"""Stage 4b: jobs for the segment-level Silero condition 'ssil' on AMI eval (exp/mv2/queue/jobs_ssil.json).

The padding is the existing wav2vec2 FA dev optimum (exp/mv2/opt/<dev>.json: IHM 100/50, SDM 0/0); no new grid.
  ssil00   no padding          wav2vec2 / Transformer / Conformer, AMI eval IHM + SDM
  ssilopt  FA dev optimum      same, skipped where the optimum is 0/0 (SDM), since it equals ssil00
Transformer uses the same fixed sample as the main run (exp/mv2/queue/subset). Workers take this list with
--jobs exp/mv2/queue/jobs_ssil.json; outputs go to exp/mv2/dec like every other job and are scored by s5.

    python scripts/mv2/s4b_make_jobs_ssil.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import EXP, MODELS, fa_rows, load_opt, write_json
from scripts.mv2.s4_make_jobs import ROWS_PER_SHARD, est_audio

EVAL = ['ami_eval_ihm', 'ami_eval_sdm']


def main():
    jobs = []
    for s in EVAL:
        fa = fa_rows(s)
        keep = set(json.load(open(EXP / 'queue' / 'subset' / f'{s}.json')))
        conds = [('ssil00', [[0, 0]])] + ([('ssilopt', 'opt')] if load_opt(s) != (0, 0) else [])
        for m in MODELS:
            rows = [r for r in fa if r['uid'] in keep] if m == 'transformer' else fa
            n = max(1, round(len(rows) / ROWS_PER_SHARD[m]))
            for tag, cells in conds:
                for i in range(n):
                    jobs.append({'id': f'{m}__{s}__{tag}__{i:03d}of{n:03d}', 'kind': 'decode', 'model': m, 'set': s, 'cond': 'ssil',
                                 'cells': cells, 'shard': [i, n], 'deps': [], 'subset': m == 'transformer',
                                 'prio': 1 if m == 'transformer' else 2, 'est': est_audio(rows[i::n], 'fa', cells)})
    write_json(EXP / 'queue' / 'jobs_ssil.json', jobs)
    tot = {}
    for j in jobs:
        tot[j['model']] = tot.get(j['model'], 0) + j['est']
    print(f'{len(jobs)} jobs; audio to decode (h): ' + ', '.join(f'{m} {v / 3600:.1f}' for m, v in tot.items()), flush=True)


if __name__ == '__main__':
    main()
