"""Throughput of each recognizer on a small sample of base segments (audio seconds decoded per wall second).

    CUDA_VISIBLE_DEVICES=0 python scripts/mv2/bench.py --model wav2vec2 --n 120
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2 import worker
from scripts.mv2.common import EXP, SR, base_rows, load_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    res = {}
    for s in ('ls_dev_other', 'ted_dev', 'ami_dev_ihm'):
        rows = random.Random(0).sample(base_rows(s), args.n)
        items = []
        for r in rows:
            rec = load_rec(r['rec_path'])
            items.append(rec[int(r['seg_start'] * SR):int(r['seg_end'] * SR)].astype('float32'))
        worker.DECODERS[args.model](items[:4])  # warm-up
        t0 = time.time()
        worker.DECODERS[args.model](items)
        wall = time.time() - t0
        audio = sum(a.size for a in items) / SR
        res[s] = {'audio_s': audio, 'wall_s': wall, 'x_realtime': audio / wall, 'n': len(items)}
        print(f'{args.model} {s}: {len(items)} segs, {audio:.0f}s audio in {wall:.1f}s -> {audio / wall:.0f}x realtime', flush=True)
    out = EXP / 'bench' / f'{args.model}{args.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
