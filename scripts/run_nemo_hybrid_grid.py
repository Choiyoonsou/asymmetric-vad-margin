"""Margin conditions x decoder head x look-ahead on NeMo's cache-aware FastConformer hybrid.

Separates the two explanations for the Emformer trailing-margin effect: is it
the Transducer decoder, or the limited right context of a streaming encoder?
One encoder carries both a CTC and an RNN-T head and supports four look-aheads
(att_context_size [70, R] with R in {0, 1, 6, 13} = 0 / 80 / 480 / 1040 ms), so
decoder type and right context vary with everything else fixed.

The tail after the MFA end is either real audio from the source recording, or
digital silence appended after the cut. "zeros_matched" appends exactly as much
silence as the real tail actually delivered, which matters on LibriSpeech where
the file often ends a few hundred ms after the last word.

Needs the vmg-nemo conda env (NeMo is not installed in vad-margin-gen):
    $PY_NEMO scripts/run_nemo_hybrid_grid.py --split ihm
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
SAMPLE_RATE = 16000
LOOKAHEADS = {0: 0, 1: 80, 6: 480, 13: 1040}  # att_context_size right frames -> ms
HEADS = ["rnnt", "ctc"]
# (condition name, leading ms, tail kind, trailing ms)
CONDITIONS = [
    ("l000_real000", 0, "real", 0),
    ("l400_real000", 400, "real", 0),
    ("l000_real400", 0, "real", 400),
    ("l400_real050", 400, "real", 50),
    ("l400_real100", 400, "real", 100),
    ("l400_real150", 400, "real", 150),
    ("l400_real400", 400, "real", 400),
    ("l400_zeros100", 400, "zeros", 100),
    ("l400_zeros400", 400, "zeros", 400),
    ("l400_zerosmatched400", 400, "zeros_matched", 400),
]
# How much silence is enough? Leading fixed at 400 ms of real audio; the tail is
# digital silence of increasing length, with the real 400 ms tail as reference.
FLUSH_CONDITIONS = [
    ("l400_real000", 400, "real", 0),
    ("l400_zeros040", 400, "zeros", 40),
    ("l400_zeros080", 400, "zeros", 80),
    ("l400_zeros160", 400, "zeros", 160),
    ("l400_zeros240", 400, "zeros", 240),
    ("l400_zeros320", 400, "zeros", 320),
    ("l400_zeros400", 400, "zeros", 400),
    ("l400_zeros640", 400, "zeros", 640),
    ("l400_zeros1000", 400, "zeros", 1000),
    ("l400_real400", 400, "real", 400),
]
# Cutting into speech (negative leading/trailing), held-out-data style dose
# response matching the wav2vec2 axes_deep sweep: one axis cut at a time,
# depth in {-50,-100,-150,-200} ms, other axis at the exact FA boundary.
# build_variants() already handles negative lead_ms/trail_ms correctly for
# kind="real" (it shrinks the window instead of padding it).
CUT_CONDITIONS = [
    ("l000_real000", 0, "real", 0),
    ("lm050_real000", -50, "real", 0),
    ("lm100_real000", -100, "real", 0),
    ("lm150_real000", -150, "real", 0),
    ("lm200_real000", -200, "real", 0),
    ("l000_realm050", 0, "real", -50),
    ("l000_realm100", 0, "real", -100),
    ("l000_realm150", 0, "real", -150),
    ("l000_realm200", 0, "real", -200),
]
GRID36_CONDITIONS = [(f"l{L:03d}_real{T:03d}", L, "real", T) for L in range(0, 501, 100) for T in range(0, 501, 100)]
CONDITION_SETS = {"margins": CONDITIONS, "flush": FLUSH_CONDITIONS, "cuts": CUT_CONDITIONS, "grid36": GRID36_CONDITIONS}

LOGGER = logging.getLogger("run_nemo_hybrid_grid")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, help="Manifest split name, e.g. test-clean or ihm.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=300, help="Utterances held in memory at once.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conditions", choices=sorted(CONDITION_SETS), default="margins")
    parser.add_argument("--output", default=None)
    parser.add_argument("--heads", nargs="+", default=HEADS, help="Decoder heads to run (rnnt, ctc).")
    parser.add_argument("--cells", nargs="+", default=None, help="Explicit L,T real-audio margins (overrides --conditions), e.g. 450,300.")
    parser.add_argument("--grid-manifest", default=None, help="Grid manifest holding the FA windows (default artifacts/manifests/<split>_grid2d.jsonl).")
    parser.add_argument("--lookaheads", nargs="+", type=int, default=list(LOOKAHEADS.values()), help="Look-aheads (ms) to run.")
    return parser.parse_args()


def load_base_rows(split, limit, path=None):
    """One row per utterance: the L=0/T=0 entry of the grid manifest carries the MFA window."""
    path = Path(path) if path else PROJECT_ROOT / "artifacts" / "manifests" / f"{split}_grid2d.jsonl"
    rows = [json.loads(line) for line in open(path)]
    base = [r for r in rows if r["leading_ms"] == 0 and r["trailing_ms"] == 0]
    return base[:limit] if limit else base


def read_window(path, start_sec, end_sec):
    info = sf.info(path)
    sr = info.samplerate
    start = max(0, int(round(start_sec * sr)))
    stop = min(info.frames, int(round(end_sec * sr)))
    audio, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return audio


CTX = 0.5  # seconds read beyond each FA edge; covers margins up to 500 ms


def build_variants(row, conditions):
    """All margin conditions for one utterance, cut from a single read of the source."""
    t_s, t_e = row["slice_start_sec"], row["slice_end_sec"]
    audio = read_window(row["source_audio_path"], t_s - CTX, t_e + CTX)
    # Offsets inside `audio`; the read is clamped at the file edges, so measure them.
    lead_avail = int(round(min(CTX, t_s) * SAMPLE_RATE))
    core_end = lead_avail + int(round((t_e - t_s) * SAMPLE_RATE))
    tail_avail_samples = max(0, len(audio) - core_end)

    variants = []
    for name, lead_ms, kind, trail_ms in conditions:
        start = lead_avail - min(lead_avail, int(lead_ms * SAMPLE_RATE / 1000))
        want = int(trail_ms * SAMPLE_RATE / 1000)
        real_tail = min(want, tail_avail_samples) if kind in ("real", "zeros_matched") else 0
        if kind == "real":
            seg = audio[start:core_end + real_tail]
            tail_ms = 1000 * real_tail / SAMPLE_RATE
        elif kind == "zeros":
            seg = np.concatenate([audio[start:core_end], np.zeros(want, np.float32)])
            tail_ms = trail_ms
        else:  # zeros_matched: silence exactly as long as the real tail that was available
            seg = np.concatenate([audio[start:core_end], np.zeros(real_tail, np.float32)])
            tail_ms = 1000 * real_tail / SAMPLE_RATE
        variants.append((name, lead_ms, kind, trail_ms, round(tail_ms, 1), seg))
    return variants


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    import nemo.collections.asr as nemo_asr

    conditions = CONDITION_SETS[args.conditions]
    if args.cells:
        conditions = [(f"l{L:03d}_real{T:03d}", L, "real", T) for L, T in (tuple(int(v) for v in c.split(",")) for c in args.cells)]
    suffix = "" if args.conditions == "margins" else f"_{args.conditions}"
    output = Path(args.output) if args.output else PROJECT_ROOT / "exp" / "nemo_hybrid" / "decodes" / f"{args.split}{suffix}__fastconformer_hybrid.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_base_rows(args.split, args.limit, args.grid_manifest)
    LOGGER.info("START split=%s utterances=%d conditions=%d output=%s", args.split, len(rows), len(conditions), output)

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME).cuda().eval()
    started = time.time()
    with open(output, "w") as fh:
        for chunk_start in range(0, len(rows), args.chunk):
            chunk = rows[chunk_start:chunk_start + args.chunk]
            items = []
            for row in chunk:
                for name, lead_ms, kind, trail_ms, tail_ms, seg in build_variants(row, conditions):
                    items.append((row, name, lead_ms, kind, trail_ms, tail_ms, seg))
            order = sorted(range(len(items)), key=lambda i: len(items[i][-1]))
            audio = [items[i][-1] for i in order]
            for head in args.heads:
                model.change_decoding_strategy(decoder_type=head, verbose=False)
                for right, look_ms in LOOKAHEADS.items():
                    if look_ms not in args.lookaheads:
                        continue
                    model.encoder.set_default_att_context_size([70, right])
                    with torch.inference_mode():
                        hyps = model.transcribe(audio, batch_size=args.batch_size, verbose=False)
                    for i, hyp in zip(order, hyps):
                        row, name, lead_ms, kind, trail_ms, tail_ms, seg = items[i]
                        fh.write(json.dumps({
                            "source_utterance_id": row["source_utterance_id"], "split": args.split,
                            "condition": name, "leading_ms": lead_ms, "tail_kind": kind,
                            "trailing_ms": trail_ms, "tail_ms_actual": tail_ms,
                            "decoded_duration_sec": round(len(seg) / SAMPLE_RATE, 3),
                            "decoder": head, "lookahead_ms": look_ms,
                            "hypothesis": hyp.text if hasattr(hyp, "text") else hyp,
                            "reference": row.get("reference") or row["transcript_reference"],
                            "model_name": "fastconformer-hybrid-streaming",
                        }) + "\n")
            fh.flush()
            LOGGER.info("progress split=%s %d/%d utterances (%.0fs)", args.split,
                        min(chunk_start + args.chunk, len(rows)), len(rows), time.time() - started)
    LOGGER.info("END split=%s output=%s", args.split, output)


if __name__ == "__main__":
    main()
