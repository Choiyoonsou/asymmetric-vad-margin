"""Stage 7b: S / D / I of the FA condition only (Part 1), from exp/mv2/results/segments.csv.gz.

  every FA cell decoded: dev + eval, three models (FA 0/0, FA + dev optimum), the wav2vec2 dev grid (121 margins)
  and the wav2vec2 cuts (-50 / -100 ms at either edge). Transformer cells are on its fixed sample.
Same counters as s7_sid.py (character S/D/I, word S/D/I, first / last word errors, edge insertions, empty).

Writes exp/mv2/results/sid_fa.json  {"set|model|L|T": {...}}.

    python scripts/mv2/s7b_sid_fa.py
"""
import csv
import gzip
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import EXP, write_json
from scripts.mv2.s7_sid import FIELDS, run

RES = EXP / 'results'


def main():
    rows = []
    with gzip.open(RES / 'segments.csv.gz', 'rt') as fh:
        for r in csv.DictReader(fh):
            if r['cond'] != 'fa' or not r['ref']:
                continue
            rows.append(((r['set'], r['model'], r['L_req'], r['T_req'], r['uid']), r['ref'], r['hyp']))
    # the same (set, model, L, T, uid) can come from two jobs (e.g. FA 0/0 and an optimum of 0/0): keep one
    uniq = {k: (k[:4], ref, hyp) for k, ref, hyp in rows}
    rows = list(uniq.values())
    chunks = [rows[i::384] for i in range(384)]
    tot = defaultdict(lambda: [0] * len(FIELDS))
    with ProcessPoolExecutor(64) as ex:
        for part in ex.map(run, chunks, chunksize=2):
            for k, v in part.items():
                t = tot[k]
                for i, x in enumerate(v):
                    t[i] += x
    write_json(RES / 'sid_fa.json', {'|'.join(k): dict(zip(FIELDS, v)) for k, v in tot.items()})
    print(f'{len(rows)} rows, {len(tot)} cells', flush=True)


if __name__ == '__main__':
    main()
