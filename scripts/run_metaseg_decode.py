"""Decode corpus-metadata segments (no VAD, no margin) and time each decode.

Latency here is the recognizer's processing time once the whole segment is in
hand: GPU, batch 1, audio already in memory, bracketed by cuda.synchronize.

  wav2vec2     -- every utterance decoded at batch 1, so every row carries proc_ms
  transformer  -- hypotheses from a length-sorted batched pass (beam search + LM is
                  slow), plus batch-1 timing on a fixed random sample per set

    python scripts/run_metaseg_decode.py --set ls_test_clean --model wav2vec2
"""
import argparse
import json
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

from src.asr import AsrSettings, BACKBONE_REGISTRY, SpeechBrainEncDecRunner, Wav2Vec2Runner, decode_tokens

SR = 16000
MODELS = {"wav2vec2": "wav2vec2-base-960h", "transformer": "speechbrain-transformer"}


def timing_sample(rows, n, seed=0):
    ids = sorted(r["utterance_id"] for r in rows)
    return set(random.Random(seed).sample(ids, min(n, len(ids))))


def load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SR).numpy()
    return audio


def timed(fn, device):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return out, 1000 * (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-sec", type=float, default=120.0, help="Cap on padded audio per transformer batch.")
    p.add_argument("--timing-n", type=int, default=300)
    p.add_argument("--shard", default=None, help="k/n: batched decode of every n-th pending utterance only (transformer).")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(PROJECT_ROOT / "artifacts" / "manifests" / "metaseg" / f"{args.set}.jsonl")]
    out = PROJECT_ROOT / "exp" / "metaseg" / "decodes" / f"{args.set}__{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    name = MODELS[args.model]
    settings = AsrSettings(model_name=name, hf_id=BACKBONE_REGISTRY[name]["hf_id"], device="cuda", device_index=0)
    device = "cuda:0"
    sample = timing_sample(rows, args.timing_n)
    hyps, proc = {}, {}
    started = time.time()

    if args.model == "wav2vec2":
        runner = Wav2Vec2Runner(settings)
        model = runner.model

        def run(audio):
            with torch.inference_mode():
                emissions, _ = model(torch.from_numpy(audio).unsqueeze(0).to(device))
            return decode_tokens(torch.argmax(emissions, dim=-1)[0].cpu().tolist(), runner.labels).lower()

        run(load(rows[0]["audio_path"]))  # warm-up
        for k, r in enumerate(rows):
            a = load(r["audio_path"])
            hyps[r["utterance_id"]], proc[r["utterance_id"]] = timed(lambda: run(a), device)
            if k % 500 == 0:
                print(f"{args.set} wav2vec2 {k}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    else:
        runner = SpeechBrainEncDecRunner(settings)
        model = runner.model

        def run_batch(audios):
            n = max(len(a) for a in audios)
            batch = torch.zeros(len(audios), n)
            rel = torch.zeros(len(audios))
            for i, a in enumerate(audios):
                batch[i, : len(a)] = torch.from_numpy(a)
                rel[i] = len(a) / n
            with torch.inference_mode():
                h, _ = model.transcribe_batch(batch.to(device), rel.to(device))
            return [x.strip().lower() for x in h]

        # Batches are capped by padded audio length as well as count: long TED/AMI
        # segments at batch 16 run the T4 out of memory.
        # Batched hypotheses are appended to a .partial file so a crash resumes instead of restarting.
        # With --shard k/n a process decodes only every n-th pending utterance into its own
        # .partial<k> file and stops; a final unsharded run merges them and does the timing.
        for f in out.parent.glob(out.stem + ".partial*"):
            for line in open(f):
                x = json.loads(line)
                hyps[x["utterance_id"]] = x["hypothesis"]
        shard_k, shard_n = (int(v) for v in args.shard.split("/")) if args.shard else (0, 1)
        pfh = open(out.with_suffix(f".partial{shard_k}" if args.shard else ".partial"), "a")
        order = sorted((r for r in rows if r["utterance_id"] not in hyps), key=lambda r: r["duration_sec"])
        order = order[shard_k::shard_n]
        chunks, cur = [], []
        for r in order:
            if cur and (len(cur) >= args.batch_size or (len(cur) + 1) * r["duration_sec"] > args.max_batch_sec):
                chunks.append(cur)
                cur = []
            cur.append(r)
        chunks.append(cur)
        cpu_model = []

        def run_cpu(a):
            # Last resort for a single segment too long for the T4: decode it on CPU.
            if not cpu_model:
                cpu_model.append(SpeechBrainEncDecRunner(AsrSettings(model_name=name, hf_id=settings.hf_id,
                                                                     device="cpu", device_index=0)).model)
            with torch.inference_mode():
                h, _ = cpu_model[0].transcribe_batch(torch.from_numpy(a).unsqueeze(0), torch.tensor([1.0]))
            print(f"{args.set} transformer CPU fallback for a {len(a) / SR:.1f}s segment", flush=True)
            return [h[0].strip().lower()]

        def run_safe(audios):
            # On OOM, split the batch in half and retry rather than lose the run.
            try:
                return run_batch(audios)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(audios) == 1:
                    return run_cpu(audios[0])
                half = len(audios) // 2
                return run_safe(audios[:half]) + run_safe(audios[half:])

        done = 0
        for k, chunk in enumerate(chunks):
            for r, h in zip(chunk, run_safe([load(r["audio_path"]) for r in chunk])):
                hyps[r["utterance_id"]] = h
                pfh.write(json.dumps({"utterance_id": r["utterance_id"], "hypothesis": h}) + "\n")
            pfh.flush()
            done += len(chunk)
            if k % 50 == 0:
                print(f"{args.set} transformer batched {done}/{len(rows)} {time.time() - started:.0f}s", flush=True)
        if args.shard:
            print(f"SHARD_DONE {args.set} {args.shard} {done} utts {time.time() - started:.0f}s", flush=True)
            return
        timed_rows = [r for r in rows if r["utterance_id"] in sample]
        run_batch([load(timed_rows[0]["audio_path"])])  # warm-up
        skipped = 0
        for r in timed_rows:
            a = load(r["audio_path"])
            try:
                _, proc[r["utterance_id"]] = timed(lambda: run_batch([a]), device)
            except torch.OutOfMemoryError:  # too long for GPU at batch 1: left out of the latency sample
                torch.cuda.empty_cache()
                skipped += 1
        if skipped:
            print(f"{args.set} transformer timing skipped {skipped} segment(s) that do not fit the GPU", flush=True)

    with open(out, "w") as fh:
        for r in rows:
            u = r["utterance_id"]
            fh.write(json.dumps({"utterance_id": u, "set": args.set, "model": args.model,
                                 "reference": r["transcript"], "hypothesis": hyps[u],
                                 "duration_sec": r["duration_sec"], "proc_ms": proc.get(u),
                                 "timing_sample": u in sample}) + "\n")
    print(f"DONE {args.set} {args.model} {len(rows)} utts {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
