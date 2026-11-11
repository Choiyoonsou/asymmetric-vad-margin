"""Stage 4: write the job list (exp/mv2/queue/jobs.json) that the GPU workers pull from.

Per set and model (conditions the instructions ask for, plus FA 0/0 as the margin-0 reference):
  base     metadata segment                           all models, dev + eval
  sil0     Silero, no padding                         all models, dev + eval
  fa_grid  FA + 121 margins (sharded)                 wav2vec2, dev only -> 'select' job picks the optimum
  fa00     FA, margin 0/0                             Transformer / Conformer (dev + eval), wav2vec2 eval
  faopt    FA + dev optimum                           Transformer / Conformer (dev + eval), wav2vec2 eval
  silopt   Silero + dev optimum                       all models, dev + eval
  facut    FA with the speech cut by 50 / 100 ms at the start (L < 0) or at the end (T < 0)
                                                      wav2vec2, dev + eval (does the leading edge matter more?)
Jobs that need an optimum depend on 'opt:<dev set>'; eval sets use the optimum of their dev set.
Priority: the grid first (it unlocks everything that needs an optimum), then the long Transformer jobs.

    python scripts/mv2/s4_make_jobs.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import DEV_SETS, EXP, GRID, MODELS, SETS, base_rows, dev_of, fa_rows, write_json

CUT_CELLS = [[-50, 0], [-100, 0], [0, -50], [0, -100]]
ROWS_PER_SHARD = {'wav2vec2': 1500, 'transformer': 125, 'conformer': 4000}
TR_SAMPLE = 500  # Transformer: fixed random sample of FA-aligned segments per set (seed 0), same for every condition
GRID_ROWS_PER_SHARD = 120


def est_audio(rows, cond, cells):
    """Audio seconds a job will decode (used for scheduling and the ETA)."""
    if cond == 'base':
        return sum(r['dur'] for r in rows)
    span = sum(r['fa_end'] - r['fa_start'] for r in rows)
    if cells == 'opt':
        return span + 0.5 * len(rows)
    return sum(span + (L + T) / 1000.0 * len(rows) for L, T in cells)


def main():
    jobs = []
    for s in SETS:
        fa = fa_rows(s)
        base = base_rows(s)
        dev = dev_of(s)
        if s in DEV_SETS:
            n = max(1, round(len(fa) / GRID_ROWS_PER_SHARD))
            grid_ids = []
            for i in range(n):
                jid = f'wav2vec2__{s}__fa_grid__{i:03d}of{n:03d}'
                grid_ids.append(jid)
                jobs.append({'id': jid, 'kind': 'decode', 'model': 'wav2vec2', 'set': s, 'cond': 'fa', 'cells': 'grid',
                             'shard': [i, n], 'deps': [], 'prio': 0, 'est': est_audio(fa[i::n], 'fa', GRID)})
            jobs.append({'id': f'select__{s}', 'kind': 'select', 'model': 'select', 'set': s, 'cond': 'select', 'cells': None,
                         'shard': [0, 1], 'deps': grid_ids, 'prio': 0, 'est': 0.0})
        sample = sorted(random.Random(0).sample(sorted(r['uid'] for r in fa), min(TR_SAMPLE, len(fa))))
        write_json(EXP / 'queue' / 'subset' / f'{s}.json', sample)
        keep = set(sample)
        for m in MODELS:
            conds = [('base', None, []), ('sil', [[0, 0]], []), ('sil', 'opt', [f'opt:{dev}'])]
            if not (m == 'wav2vec2' and s in DEV_SETS):
                conds += [('fa', [[0, 0]], []), ('fa', 'opt', [f'opt:{dev}'])]
            for cond, cells, deps in conds:
                rows = base if cond == 'base' else fa
                if m == 'transformer':
                    rows = [r for r in rows if r['uid'] in keep]
                n = max(1, round(len(rows) / ROWS_PER_SHARD[m]))
                tag = cond if cond == 'base' else f"{cond}{'opt' if cells == 'opt' else '00'}"
                for i in range(n):
                    jobs.append({'id': f'{m}__{s}__{tag}__{i:03d}of{n:03d}', 'kind': 'decode', 'model': m, 'set': s, 'cond': cond,
                                 'cells': cells, 'shard': [i, n], 'deps': deps, 'subset': m == 'transformer', 'prio': 1 if m == 'transformer' else 2,
                                 'est': est_audio(rows[i::n], cond, cells)})
        n = max(1, round(len(fa) / 2000))
        for i in range(n):
            jobs.append({'id': f'wav2vec2__{s}__facut__{i:03d}of{n:03d}', 'kind': 'decode', 'model': 'wav2vec2', 'set': s, 'cond': 'fa',
                         'cells': CUT_CELLS, 'shard': [i, n], 'deps': [], 'prio': 0, 'est': est_audio(fa[i::n], 'fa', CUT_CELLS)})
    write_json(EXP / 'queue' / 'jobs.json', jobs)
    tot = {}
    for j in jobs:
        tot[j['model']] = tot.get(j['model'], 0) + j['est']
    print(f'{len(jobs)} jobs; audio to decode (h): ' + ', '.join(f'{m} {v / 3600:.0f}' for m, v in tot.items()), flush=True)


if __name__ == '__main__':
    main()
