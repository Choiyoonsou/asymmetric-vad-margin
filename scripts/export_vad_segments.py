"""Stage 1 of the NeMo VAD pipeline: run a VAD once and save its raw segments.

NeMo lives in its own conda env (vmg-nemo) without the VAD dependencies, so the
VAD-policy experiment for the FastConformer hybrid runs in two stages:

  1. this script (vad-margin-gen env) runs Silero / WebRTC / pyannote and writes
     each utterance's detected segments, as sample offsets into the VAD's
     16 kHz audio, before any margin policy;
  2. scripts/run_nemo_vad_decode.py (vmg-nemo env) applies every policy to the
     saved segments and decodes with both heads at every look-ahead.

The VADs run on CPU, so this stage does not compete with GPU decoding.

    python scripts/export_vad_segments.py --split ihm --vad silero --workers 16
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_config, read_jsonl

LOGGER = logging.getLogger("export_vad_segments")
_RUNNER = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--vad", required=True, choices=["silero", "webrtc", "pyannote"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _init(config_path, vad_name):
    import torch

    torch.set_num_threads(1)
    from src.vad import create_vad_runner

    global _RUNNER
    _RUNNER = create_vad_runner(load_config(config_path), vad_name, None)


def _detect(row):
    from src.audio.io import read_audio

    audio, sample_rate = read_audio(row["audio_path"])
    vad_audio, vad_rate, segments = _RUNNER.detect_segments(audio, sample_rate)
    return {
        "utterance_id": row["utterance_id"],
        "split": row["split"],
        "audio_path": row["audio_path"],
        "reference": row["transcript"],
        "vad_name": _RUNNER.name,
        "vad_lookahead_ms": _RUNNER.lookahead_ms,
        "sample_rate": int(vad_rate),
        "num_samples": int(len(vad_audio)),
        "segments": [[int(s["start"]), int(s["end"])] for s in segments],
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config(args.config)
    rows = read_jsonl(Path(config["manifest"]["output_path"]) / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.output) if args.output else PROJECT_ROOT / "exp" / "vad_segments" / f"{args.split}__{args.vad}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("START split=%s vad=%s rows=%d output=%s", args.split, args.vad, len(rows), output)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.config, args.vad)) as pool:
        results = list(pool.map(_detect, rows, chunksize=8))
    with open(output, "w") as fh:
        for result in results:
            fh.write(json.dumps(result) + "\n")
    empty = sum(1 for r in results if not r["segments"])
    LOGGER.info("END split=%s vad=%s rows=%d empty=%d output=%s", args.split, args.vad, len(results), empty, output)


if __name__ == "__main__":
    main()
