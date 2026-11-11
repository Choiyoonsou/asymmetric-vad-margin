"""E5: measured end-of-utterance latency under true cache-aware streaming.

Feeds each clip chunk by chunk through NeMo's cache-aware FastConformer hybrid
(conformer_stream_step) and records, after every chunk, how much audio has been
consumed, how long the step took on the GPU, and the running transcript. From
that it derives, relative to the MFA end of speech t_e:

  emit_audio_ms   audio position (after t_e) at which the transcript reached its
                  final form -- how far past the end of speech the model had to
                  read before it committed the last token.
  wait_ms         time spent waiting for audio after t_e. Real tail audio
                  arrives in real time, so it is waited for; appended digital
                  silence exists the moment the endpoint fires, so it is not.
  compute_ms      GPU time of the steps that ran after t_e.
  latency_ms      wait_ms + compute_ms: end of speech to final transcript,
                  excluding the VAD endpoint delay common to every condition.

Needs the vmg-nemo conda env:
    $PY_NEMO scripts/run_nemo_streaming_latency.py --split ihm --limit 200
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from run_nemo_hybrid_grid import MODEL_NAME, SAMPLE_RATE, load_base_rows, read_window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOKAHEADS = {0: 0, 1: 80, 6: 480, 13: 1040}
HEADS = ["rnnt", "ctc"]
LEADING_MS = 400
# (condition, tail kind, tail ms)
TAILS = [
    ("real000", "real", 0),
    ("real160", "real", 160),
    ("real400", "real", 400),
    ("zeros160", "zeros", 160),
    ("zeros400", "zeros", 400),
]
# Added after the E6 sweep showed 640 ms of silence saturates the gain.
TAILS_640 = [
    ("real640", "real", 640),
    ("zeros640", "zeros", 640),
]
TAIL_SETS = {"base": TAILS, "640": TAILS_640}

LOGGER = logging.getLogger("run_nemo_streaming_latency")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=200, help="Random utterances per split (seed 0).")
    parser.add_argument("--tails", choices=sorted(TAIL_SETS), default="base")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_clip(row, kind, tail_ms):
    """Leading 400 ms of real audio, the utterance, then a real or silent tail."""
    t_s, t_e = row["slice_start_sec"], row["slice_end_sec"]
    audio = read_window(row["source_audio_path"], t_s - LEADING_MS / 1000, t_e + tail_ms / 1000)
    lead = int(round(min(LEADING_MS / 1000, t_s) * SAMPLE_RATE))
    speech_end = lead + int(round((t_e - t_s) * SAMPLE_RATE))
    if kind == "real":
        clip = audio[: speech_end + int(tail_ms * SAMPLE_RATE / 1000)]
    else:
        clip = np.concatenate([audio[:speech_end], np.zeros(int(tail_ms * SAMPLE_RATE / 1000), np.float32)])
    real_tail_ms = 1000 * max(0, min(len(clip), len(audio)) - speech_end) / SAMPLE_RATE if kind == "real" else 0.0
    return clip.astype(np.float32), 1000 * speech_end / SAMPLE_RATE, real_tail_ms


def stream(model, audio):
    """Run one clip through cache-aware streaming; return per-step (audio_ms, gpu_ms, text)."""
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
    buffer.append_audio(audio, stream_id=-1)
    hop_ms = 1000 * model.cfg.preprocessor.window_stride
    total_frames = buffer.buffer.size(-1)
    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    prev_hyp, prev_pred = None, None
    steps = []
    for step, (chunk, chunk_len) in enumerate(buffer):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prev_pred, texts, cache_channel, cache_time, cache_len, prev_hyp = model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_channel,
                cache_last_time=cache_time,
                cache_last_channel_len=cache_len,
                keep_all_outputs=buffer.is_buffer_empty(),
                previous_hypotheses=prev_hyp,
                previous_pred_out=prev_pred,
                drop_extra_pre_encoded=0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
                return_transcription=True,
            )
        torch.cuda.synchronize()
        gpu_ms = 1000 * (time.perf_counter() - started)
        text = texts[0].text if hasattr(texts[0], "text") else texts[0]
        consumed_ms = hop_ms * min(buffer.buffer_idx, total_frames)
        steps.append((consumed_ms, gpu_ms, text))
    return steps


def summarize(steps, speech_end_ms, kind, real_tail_ms):
    final = steps[-1][2]
    emit_idx = next(i for i, s in enumerate(steps) if s[2] == final)
    emit_audio_ms = max(0.0, steps[emit_idx][0] - speech_end_ms)
    after = [i for i, s in enumerate(steps) if s[0] > speech_end_ms and i <= emit_idx]
    compute_ms = sum(steps[i][1] for i in after)
    wait_ms = min(emit_audio_ms, real_tail_ms) if kind == "real" else 0.0
    return final, emit_audio_ms, wait_ms, compute_ms


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    rows = load_base_rows(args.split, None)
    random.Random(0).shuffle(rows)
    rows = rows[: args.limit]
    tails = TAIL_SETS[args.tails]
    suffix = "" if args.tails == "base" else f"_{args.tails}"
    output = Path(args.output) if args.output else PROJECT_ROOT / "exp" / "nemo_streaming" / f"{args.split}{suffix}__latency.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("START split=%s utterances=%d output=%s", args.split, len(rows), output)

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME).cuda().eval()
    started = time.time()
    with open(output, "w") as fh:
        for n, row in enumerate(rows, 1):
            clips = {name: build_clip(row, kind, ms) for name, kind, ms in tails}
            for head in HEADS:
                model.change_decoding_strategy(decoder_type=head, verbose=False)
                for right, look_ms in LOOKAHEADS.items():
                    model.encoder.set_default_att_context_size([70, right])
                    model.encoder.setup_streaming_params()
                    for name, kind, ms in tails:
                        clip, speech_end_ms, real_tail_ms = clips[name]
                        steps = stream(model, clip)
                        final, emit_ms, wait_ms, compute_ms = summarize(steps, speech_end_ms, kind, real_tail_ms)
                        fh.write(json.dumps({
                            "source_utterance_id": row["source_utterance_id"], "split": args.split,
                            "decoder": head, "lookahead_ms": look_ms, "condition": f"l400_{name}",
                            "tail_kind": kind, "tail_ms": ms, "real_tail_ms": round(real_tail_ms, 1),
                            "emit_audio_ms": round(emit_ms, 1), "wait_ms": round(wait_ms, 1),
                            "compute_ms": round(compute_ms, 2), "latency_ms": round(wait_ms + compute_ms, 1),
                            "n_steps": len(steps), "hypothesis": final,
                            "reference": row.get("reference") or row["transcript_reference"],
                        }) + "\n")
            if n % 20 == 0:
                fh.flush()
                LOGGER.info("progress split=%s %d/%d (%.0fs)", args.split, n, len(rows), time.time() - started)
    LOGGER.info("END split=%s output=%s", args.split, output)


if __name__ == "__main__":
    main()
