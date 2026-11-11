"""Load each registered ASR backbone and run one forward pass on a synthetic
signal, to catch integration errors (wrong shapes, missing deps, bad HF ids)
before committing to a full data download + MFA alignment + decode run.

This does NOT validate transcription quality -- the input isn't real speech,
so hypotheses will be garbage. It only proves each backbone loads and runs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import BACKBONE_REGISTRY, RUNNER_CLASSES, AsrSettings
from src.audio.io import write_audio


def make_placeholder_audio(path, duration_sec=2.0, sample_rate=16000):
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    fundamental = 120.0
    audio = sum(0.15 / (i + 1) * np.sin(2 * np.pi * fundamental * (i + 1) * t) for i in range(6))
    audio = (audio + 0.01 * np.random.randn(len(t))).astype(np.float32)
    write_audio(path, audio, sample_rate)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(BACKBONE_REGISTRY),
        help="Subset of BACKBONE_REGISTRY keys to test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio_path = PROJECT_ROOT / "artifacts" / "smoke_test_audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    make_placeholder_audio(audio_path)

    for model_name in args.backbones:
        if model_name not in BACKBONE_REGISTRY:
            print(f"[skip] unknown backbone: {model_name}")
            continue

        backbone = BACKBONE_REGISTRY[model_name]
        runner_cls = RUNNER_CLASSES[backbone["family"]]
        settings = AsrSettings(
            model_name=model_name,
            hf_id=backbone["hf_id"],
            device=args.device,
            device_index=args.device_index,
        )

        print(f"=== {model_name} ({backbone['family']}) ===")
        try:
            runner = runner_cls(settings)
            result = runner.transcribe(str(audio_path))
            print("  OK    hypothesis=%r duration_sec=%.2f" % (result["hypothesis"], result["duration_sec"]))
        except Exception as exc:  # noqa: BLE001 -- smoke test wants to report, not raise
            print(f"  FAIL  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
