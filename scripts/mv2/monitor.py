"""Progress / ETA of the decode queue, appended to logs/mv2/progress.log every --every seconds.

ETA per model = remaining audio / audio decoded per second since that model's first finished job.

    python scripts/mv2/monitor.py --models wav2vec2 --every 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import EXP, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='wav2vec2,transformer,conformer')
    ap.add_argument('--every', type=int, default=60)
    args = ap.parse_args()
    models = args.models.split(',')
    log = ROOT / 'logs' / 'mv2' / 'progress.log'
    t_start = time.time()
    while True:
        jobs = [j for j in json.load(open(EXP / 'queue' / 'jobs.json')) if j['model'] in models]
        lines, all_done = [], True
        for m in models:
            js = [j for j in jobs if j['model'] == m]
            done = [j for j in js if (EXP / 'dec' / f"{j['id']}.jsonl").exists()]
            tot = sum(j['est'] for j in js)
            got = sum(j['est'] for j in done)
            timings = [json.load(open(p)) for p in (EXP / 'timing').glob(f'{m}__*.json')] if (EXP / 'timing').exists() else []
            if timings:
                first = min(Path(EXP / 'timing' / f"{t['id']}.json").stat().st_mtime - t['wall_s'] for t in timings)
                rate = got / max(1.0, time.time() - first)
                eta = (tot - got) / rate if rate > 0 else float('inf')
                eta_s = f'{eta / 3600:.1f} h' if eta != float('inf') else '?'
            else:
                eta_s = '?'
            all_done &= len(done) == len(js)
            lines.append(f'{m}: {len(done)}/{len(js)} jobs, {got / 3600:.0f}/{tot / 3600:.0f} h audio ({100 * got / max(tot, 1):.1f}%), ETA {eta_s}')
        opts = sorted(p.stem for p in (EXP / 'opt').glob('*.json')) if (EXP / 'opt').exists() else []
        failed = len(list((EXP / 'queue' / 'failed').glob('*.txt'))) if (EXP / 'queue' / 'failed').exists() else 0
        msg = f"[{time.strftime('%H:%M:%S')}] elapsed {(time.time() - t_start) / 60:.0f} min | " + ' | '.join(lines) + f' | optima: {opts} | failed: {failed}'
        with open(log, 'a') as fh:
            fh.write(msg + '\n')
        if all_done:
            with open(log, 'a') as fh:
                fh.write(f'[{time.strftime("%H:%M:%S")}] ALL DONE for {models}\n')
            return
        time.sleep(args.every)


if __name__ == '__main__':
    main()
