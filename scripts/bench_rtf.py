"""Measure real-time factor (RTF) for every VAD and ASR backbone in the study.

RTF = processing time / audio duration, timed per utterance with
time.perf_counter() (plus torch.cuda.synchronize() on GPU), batch size 1,
after a few untimed warm-up utterances. Audio is preloaded into memory, so
file I/O is excluded; resampling, feature extraction, the network and the
decoder (greedy / beam search / LM) are included.

    vad  stage  Silero, WebRTC, pyannote on the same input each detector sees
                in E4 (the baseline manifest audio: full LibriSpeech file,
                AMI clip with 1 s context). CPU, --threads torch threads;
                pyannote is timed on GPU as well.
    asr  stage  wav2vec2 (CTC), SpeechBrain Transformer (attention + LM),
                Emformer (streaming Transducer) on the FA 400/400 clip.
    nemo stage  FastConformer hybrid (Conformer; CTC and RNN-T heads) on the
                same clip, offline at lookahead 0 and 1040 ms. Needs the
                vmg-nemo env.

The utterance sample is drawn with --seed from the grid manifest, so every
stage and env times the same utterances.

    python scripts/bench_rtf.py --stage vad --split test-clean --threads 1
    python scripts/bench_rtf.py --stage asr --split ihm --device cuda:0
    $PY_NEMO scripts/bench_rtf.py --stage nemo --split ihm --device cuda:0
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000
MARGIN_SEC = 0.4
WARMUP = 5
NEMO_MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
NEMO_LOOKAHEADS = {0: 0, 13: 1040}  # att_context_size right frames -> ms

LOGGER = logging.getLogger("bench_rtf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["vad", "asr", "nemo"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu (asr/nemo stages).")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads.")
    parser.add_argument("--models", default=None, help="Comma list to run a subset of the stage's models.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_window(path, start_sec=None, end_sec=None):
    info = sf.info(path)
    sr = info.samplerate
    start = 0 if start_sec is None else max(0, int(round(start_sec * sr)))
    stop = info.frames if end_sec is None else min(info.frames, int(round(end_sec * sr)))
    audio, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def sample_rows(split, limit, seed):
    """The FA (L=0/T=0) row of each utterance, joined to the baseline manifest audio."""
    grid = PROJECT_ROOT / "artifacts" / "manifests" / f"{split}_grid2d.jsonl"
    base = [json.loads(l) for l in open(grid)]
    base = [r for r in base if r["leading_ms"] == 0 and r["trailing_ms"] == 0]
    full = {}
    for line in open(PROJECT_ROOT / "artifacts" / "manifests" / f"{split}.jsonl"):
        r = json.loads(line)
        full[r["utterance_id"]] = r["audio_path"]
    base = [r for r in base if r["source_utterance_id"] in full]
    rng = random.Random(seed)
    rows = rng.sample(base, min(limit, len(base)))
    for r in rows:
        r["vad_audio_path"] = full[r["source_utterance_id"]]
    return rows


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def time_items(name, device, items, fn, meta, fh):
    """items: list of (row, audio). fn(audio) -> hypothesis or None."""
    for _, audio in items[:WARMUP]:
        fn(audio)
    sync(device)
    total_audio = total_proc = 0.0
    for row, audio in items:
        sync(device)
        started = time.perf_counter()
        hyp = fn(audio)
        sync(device)
        proc = time.perf_counter() - started
        audio_sec = len(audio) / SAMPLE_RATE
        total_audio += audio_sec
        total_proc += proc
        fh.write(json.dumps({
            **meta, "model": name, "device": device,
            "source_utterance_id": row["source_utterance_id"],
            "audio_sec": round(audio_sec, 4), "proc_ms": round(1000 * proc, 3),
            "rtf": round(proc / audio_sec, 6) if audio_sec else None,
            "hypothesis": hyp if isinstance(hyp, str) else None,
            "reference": row.get("transcript_reference") or row.get("reference"),
        }) + "\n")
    fh.flush()
    LOGGER.info("%-28s %-6s n=%d audio=%.1fs proc=%.2fs RTF=%.4f", name, device, len(items),
                total_audio, total_proc, total_proc / total_audio)


def run_vad(args, rows, fh, meta):
    from src.common import load_config
    from src.vad import create_vad_runner

    config = load_config(str(PROJECT_ROOT / "conf" / "base.yaml"))
    items = [(r, read_window(r["vad_audio_path"])) for r in rows]
    wanted = args.models.split(",") if args.models else ["silero", "webrtc", "pyannote_cpu", "pyannote_gpu"]
    for name in wanted:
        if name.startswith("pyannote"):
            device = "cuda:0" if name.endswith("gpu") else "cpu"
            runner = create_vad_runner(config, "pyannote", overrides=None)
            if device != "cpu":
                runner.pipeline.to(torch.device(device))
        else:
            device = "cpu"
            runner = create_vad_runner(config, name)
        fn = lambda audio, r=runner: (r.detect_segments(audio, SAMPLE_RATE), None)[1]
        time_items(name, device, items, fn, meta, fh)
        del runner


def asr_items(rows):
    return [(r, read_window(r["source_audio_path"], max(0.0, r["slice_start_sec"] - MARGIN_SEC),
                            r["slice_end_sec"] + MARGIN_SEC)) for r in rows]


def run_asr(args, rows, fh, meta):
    from src.asr import BACKBONE_REGISTRY, RUNNER_CLASSES, AsrSettings
    from src.audio.io import AudioArray

    items = asr_items(rows)
    device = args.device
    wanted = args.models.split(",") if args.models else ["wav2vec2-base-960h", "speechbrain-transformer", "emformer-rnnt-base"]
    for model_name in wanted:
        backbone = BACKBONE_REGISTRY[model_name]
        settings = AsrSettings(
            model_name=model_name, hf_id=backbone["hf_id"],
            device="cpu" if device == "cpu" else "cuda",
            device_index=0 if device == "cpu" else int(device.split(":")[1]),
        )
        runner = RUNNER_CLASSES[backbone["family"]](settings)
        fn = lambda audio, r=runner: r.transcribe(AudioArray(samples=audio, sample_rate=SAMPLE_RATE))["hypothesis"]
        time_items(model_name, device, items, fn, meta, fh)
        del runner
        if device != "cpu":
            torch.cuda.empty_cache()


def run_nemo(args, rows, fh, meta):
    import nemo.collections.asr as nemo_asr

    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    items = asr_items(rows)
    device = args.device
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(NEMO_MODEL, map_location=device).to(device).eval()

    def text_of(hyps):
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        hyp = hyps[0]
        return hyp.text if hasattr(hyp, "text") else str(hyp)

    for head in ["ctc", "rnnt"]:
        model.change_decoding_strategy(decoder_type=head, verbose=False)
        for right, look_ms in NEMO_LOOKAHEADS.items():
            model.encoder.set_default_att_context_size([70, right])

            def fn(audio, head=head):
                with torch.inference_mode():
                    sig = torch.from_numpy(audio).unsqueeze(0).to(device)
                    length = torch.tensor([sig.shape[1]], device=device)
                    feats, feat_len = model.preprocessor(input_signal=sig, length=length)
                    enc, enc_len = model.encoder(audio_signal=feats, length=feat_len)
                    if head == "ctc":
                        logp = model.ctc_decoder(encoder_output=enc)
                        hyps = model.ctc_decoding.ctc_decoder_predictions_tensor(logp, decoder_lengths=enc_len, return_hypotheses=False)
                    else:
                        hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len)
                return text_of(hyps)

            time_items(f"fastconformer-{head}-la{look_ms}", device, items, fn, meta, fh)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    torch.set_num_threads(args.threads)
    rows = sample_rows(args.split, args.limit, args.seed)
    dev_tag = args.device.replace(":", "")
    tag = f"{args.stage}_{args.split}_{'cpu' if args.stage == 'vad' else dev_tag}_t{args.threads}"
    if args.models:
        tag += "_" + args.models.replace(",", "+")
    output = Path(args.output) if args.output else PROJECT_ROOT / "exp" / "rtf" / f"{tag}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {"stage": args.stage, "split": args.split, "threads": args.threads,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    LOGGER.info("START stage=%s split=%s n=%d threads=%d output=%s", args.stage, args.split, len(rows), args.threads, output)
    with open(output, "w") as fh:
        {"vad": run_vad, "asr": run_asr, "nemo": run_nemo}[args.stage](args, rows, fh, meta)
    LOGGER.info("END output=%s", output)


if __name__ == "__main__":
    main()
