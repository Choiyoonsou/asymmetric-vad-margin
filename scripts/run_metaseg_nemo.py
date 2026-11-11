"""Decode corpus-metadata segments with NeMo's FastConformer hybrid (no VAD, no margin), and time it.

Hypotheses come from the batched model.transcribe() pass, per head x look-ahead.
Latency is timed separately at batch 1 on the same fixed random sample that
scripts/run_metaseg_decode.py uses: preprocessor + encoder + greedy decode on a
segment already in GPU memory, bracketed by cuda.synchronize.

Needs the vmg-nemo conda env:
    $PY_NEMO scripts/run_metaseg_nemo.py --set ls_test_clean
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
SR = 16000
CONFIGS = [("ctc", 13, 1040), ("rnnt", 13, 1040), ("ctc", 0, 0), ("rnnt", 0, 0)]


def load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SR).numpy()
    return audio


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--timing-n", type=int, default=300)
    args = p.parse_args()
    for noisy in ("nemo_logger", "nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    rows = [json.loads(l) for l in open(PROJECT_ROOT / "artifacts" / "manifests" / "metaseg" / f"{args.set}.jsonl")]
    ids = sorted(r["utterance_id"] for r in rows)
    sample = set(random.Random(0).sample(ids, min(args.timing_n, len(ids))))
    out = PROJECT_ROOT / "exp" / "metaseg" / "decodes" / f"{args.set}__fastconformer.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME).cuda().eval()
    started = time.time()

    order = sorted(range(len(rows)), key=lambda i: rows[i]["duration_sec"])
    timed_rows = [r for r in rows if r["utterance_id"] in sample]
    timed_audio = [load(r["audio_path"]) for r in timed_rows]
    with open(out, "w") as fh:
        for head, right, look_ms in CONFIGS:
            model.change_decoding_strategy(decoder_type=head, verbose=False)
            model.encoder.set_default_att_context_size([70, right])
            hyps = {}
            for k in range(0, len(order), 2000):
                idx = order[k:k + 2000]
                with torch.inference_mode():
                    out_h = model.transcribe([load(rows[i]["audio_path"]) for i in idx], batch_size=args.batch_size, verbose=False)
                for i, h in zip(idx, out_h):
                    hyps[rows[i]["utterance_id"]] = (h.text if hasattr(h, "text") else h).strip().lower()
            proc = {}
            for warm in range(2):  # first pass warms up kernels for this config
                for r, a in zip(timed_rows if warm else timed_rows[:3], timed_audio if warm else timed_audio[:3]):
                    with torch.inference_mode():
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        sig = torch.from_numpy(a)[None].cuda()
                        ln = torch.tensor([len(a)], device="cuda")
                        feats, flen = model.preprocessor(input_signal=sig, length=ln)
                        enc, elen = model.encoder(audio_signal=feats, length=flen)
                        if head == "ctc":
                            model.ctc_decoding.ctc_decoder_predictions_tensor(model.ctc_decoder(encoder_output=enc), decoder_lengths=elen)
                        else:
                            model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=elen)
                        torch.cuda.synchronize()
                    if warm:
                        proc[r["utterance_id"]] = 1000 * (time.perf_counter() - t0)
            for r in rows:
                u = r["utterance_id"]
                fh.write(json.dumps({"utterance_id": u, "set": args.set, "model": f"fastconformer-{head}",
                                     "lookahead_ms": look_ms, "reference": r["transcript"], "hypothesis": hyps[u],
                                     "duration_sec": r["duration_sec"], "proc_ms": proc.get(u),
                                     "timing_sample": u in sample}) + "\n")
            fh.flush()
            print(f"{args.set} {head} la{look_ms} done {time.time() - started:.0f}s", flush=True)
    print(f"DONE {args.set} fastconformer {len(rows)} utts -> {out}", flush=True)


if __name__ == "__main__":
    main()
