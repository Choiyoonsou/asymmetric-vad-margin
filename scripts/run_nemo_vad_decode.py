"""Stage 2 of the NeMo VAD pipeline: VAD policies on the FastConformer hybrid.

Reads the raw VAD segments saved by scripts/export_vad_segments.py, applies
every margin policy with the same code the other backbones use
(src/vad_policy.py: apply_vad_policy + append_trailing_silence), and decodes
the result with both decoder heads at every look-ahead. An utterance the VAD
found no speech in is recorded with an empty hypothesis.

Needs the vmg-nemo conda env:
    $PY_NEMO scripts/run_nemo_vad_decode.py --split ihm --vad silero
"""
import argparse
import re
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import concatenate_segments, read_audio
from src.common import load_config
from src.vad_policy import VadPolicySettings, algorithmic_latency_ms, append_trailing_silence, apply_vad_policy, resolve_policy

MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
LOOKAHEADS = {0: 0, 1: 80, 6: 480, 13: 1040}
HEADS = ["rnnt", "ctc"]
POLICIES = ["default", "sym_400", "asym_400_050", "asym_400_100", "asym_400_150", "flush_400", "flush_400_s640"]

LOGGER = logging.getLogger("run_nemo_vad_decode")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--vad", required=True, choices=["silero", "webrtc", "pyannote"])
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--heads", nargs="+", default=HEADS)
    parser.add_argument("--lookaheads", nargs="+", type=int, default=list(LOOKAHEADS.values()))
    return parser.parse_args()


def reconstruct(record, settings):
    """Policy-adjusted audio for one utterance, exactly as run_vad_decode builds it."""
    audio, sample_rate = read_audio(record["audio_path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != record["sample_rate"]:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sample_rate, record["sample_rate"]).numpy()
    audio = audio.astype(np.float32)[: record["num_samples"]]
    rate = record["sample_rate"]
    segments = [{"start": s, "end": e} for s, e in record["segments"]]
    if settings.leading_pad_ms or settings.trailing_pad_ms:
        segments = apply_vad_policy(segments=segments, sample_rate=rate, audio_length=len(audio), settings=settings)
    pieces = [audio[s["start"]:s["end"]] for s in segments]
    rebuilt = concatenate_segments(pieces) if pieces else np.zeros(0, np.float32)
    return append_trailing_silence(np.asarray(rebuilt, np.float32), rate, settings)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    config = load_config(args.config)
    speech_pad_ms = config["silero_vad"]["speech_pad_ms"]
    seg_path = PROJECT_ROOT / "exp" / "vad_segments" / f"{args.split}__{args.vad}.jsonl"
    records = [json.loads(line) for line in open(seg_path)]
    if args.limit:
        records = records[: args.limit]
    output = Path(args.output) if args.output else PROJECT_ROOT / "exp" / "nemo_vad" / "decodes" / f"{args.split}__{args.vad}__fastconformer_hybrid.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    def policy(name):
        # "grid_lLLL_tTTT": an ad-hoc leading/trailing margin pair for the grid sweep
        m = re.fullmatch(r"grid_l(\d+)_t(\d+)", name)
        if m:
            return VadPolicySettings(policy_name=name, leading_pad_ms=int(m.group(1)), trailing_pad_ms=int(m.group(2)))
        return resolve_policy(config, name, args.split)

    settings = {name: policy(name) for name in args.policies}
    LOGGER.info("START split=%s vad=%s utterances=%d policies=%s output=%s", args.split, args.vad, len(records), args.policies, output)

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME).cuda().eval()
    started = time.time()
    with open(output, "w") as fh:
        for chunk_start in range(0, len(records), args.chunk):
            items = []
            for record in records[chunk_start:chunk_start + args.chunk]:
                for name, policy in settings.items():
                    items.append((record, name, policy, reconstruct(record, policy)))
            decodable = sorted((i for i, it in enumerate(items) if it[3].size), key=lambda i: items[i][3].size)
            audio = [items[i][3] for i in decodable]
            for head in args.heads:
                model.change_decoding_strategy(decoder_type=head, verbose=False)
                for right, look_ms in LOOKAHEADS.items():
                    if look_ms not in args.lookaheads:
                        continue
                    model.encoder.set_default_att_context_size([70, right])
                    with torch.inference_mode():
                        hyps = model.transcribe(audio, batch_size=args.batch_size, verbose=False) if audio else []
                    text = {i: (h.text if hasattr(h, "text") else h) for i, h in zip(decodable, hyps)}
                    for i, (record, name, policy, clip) in enumerate(items):
                        fh.write(json.dumps({
                            "source_utterance_id": record["utterance_id"], "split": args.split,
                            "vad_name": args.vad, "policy": name, "condition": f"{args.vad}_policy_{name}",
                            "leading_pad_ms": policy.leading_pad_ms, "trailing_pad_ms": policy.trailing_pad_ms,
                            "trailing_silence_ms": policy.trailing_silence_ms,
                            "algorithmic_latency_ms": algorithmic_latency_ms(config, policy, record["vad_lookahead_ms"], speech_pad_ms),
                            "decoder": head, "lookahead_ms": look_ms, "vad_empty": clip.size == 0,
                            "decoded_duration_sec": round(clip.size / record["sample_rate"], 3),
                            "hypothesis": text.get(i, ""), "reference": record["reference"],
                        }) + "\n")
            fh.flush()
            LOGGER.info("progress split=%s vad=%s %d/%d (%.0fs)", args.split, args.vad,
                        min(chunk_start + args.chunk, len(records)), len(records), time.time() - started)
    LOGGER.info("END split=%s vad=%s output=%s", args.split, args.vad, output)


if __name__ == "__main__":
    main()
