"""Stage 5: score every decode with one text normalization (scripts/summarize_metaseg.norm).

Evaluation set per data set = base segments that passed the 30 s filter and have an FA span (identical for
every condition and margin). Transformer rows are scored on its fixed sample (exp/mv2/queue/subset), and
any comparison involving the Transformer uses those segments for every condition.

Writes
  exp/mv2/results/segments.csv.gz   one row per segment x model x condition x margin
                                    (requested / realized margins, errors, CER, STT-input latency)
  exp/mv2/results/agg.json          corpus CER and latency per set x model x condition x margin,
                                    plus paired-bootstrap 95% intervals for the condition comparisons
"""
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import editdistance
import numpy as np

from scripts.mv2.common import EXP, SETS, fa_rows, jl, write_json
from scripts.summarize_metaseg import norm

RES = EXP / 'results'
COLS = ['set', 'role', 'model', 'cond', 'L_req', 'T_req', 'uid', 'real_L_ms', 'real_T_ms', 'ref_chars', 'errors', 'cer',
        'lat_pre_ms', 'lat_pre_req_ms', 'dec_sec', 'n_vad', 'ref', 'hyp']


def score_file(path):
    rows = jl(path)
    if not rows:
        return path, []
    s = rows[0]['set']
    fa = {r['uid']: r for r in fa_rows(s)}
    out = []
    for x in rows:
        r = fa.get(x['uid'])
        if r is None:  # FA failed: not in the evaluation set
            continue
        ref, hyp = norm(r['transcript']), norm(x['hyp'])
        e = editdistance.eval(ref, hyp)
        if x['cond'] == 'base':  # implicit margins of the metadata segment around the FA span
            x['real_L'] = 1000.0 * (r['fa_start'] - r['seg_start'])
            x['real_T'] = x['lat_ms'] = x['lat_req_ms'] = 1000.0 * (r['seg_end'] - r['fa_end'])
        out.append([s, SETS[s]['role'], x['model'], x['cond'], x['L'], x['T'], x['uid'], x.get('real_L'), x.get('real_T'),
                    len(ref), e, round(100.0 * e / max(1, len(ref)), 3), x.get('lat_ms'), x.get('lat_req_ms'), x['dec_sec'],
                    x.get('n_vad'), ref, hyp])
    return path, out


def main():
    t0 = time.time()
    files = sorted((EXP / 'dec').glob('*.jsonl'))
    RES.mkdir(parents=True, exist_ok=True)
    subset = {s: set(json.load(open(EXP / 'queue' / 'subset' / f'{s}.json'))) for s in SETS}
    # per (set, model, cond, L, T): uid -> (errors, ref_chars, latency, realized L, realized T)
    per = {}
    with gzip.open(RES / 'segments.csv.gz', 'wt', newline='') as fh, ProcessPoolExecutor(32) as ex:
        w = csv.writer(fh)
        w.writerow(COLS)
        for k, (path, rows) in enumerate(ex.map(score_file, files, chunksize=1)):
            for row in rows:
                w.writerow(row)
                key = (row[0], row[2], row[3], row[4], row[5])
                per.setdefault(key, {})[row[6]] = (row[10], row[9], row[12], row[7], row[8])
            if (k + 1) % 50 == 0:
                print(f'scored {k + 1}/{len(files)} files ({time.time() - t0:.0f}s)', flush=True)

    def summarize(d, uids):
        uids = [u for u in uids if u in d]
        if not uids:
            return None
        e = np.array([d[u][0] for u in uids], float)
        n = np.array([d[u][1] for u in uids], float)
        lat = np.array([d[u][2] for u in uids if d[u][2] is not None], float)
        rl = np.array([d[u][3] for u in uids if d[u][3] is not None], float)
        rt = np.array([d[u][4] for u in uids if d[u][4] is not None], float)
        q = lambda a: None if not len(a) else {'mean': float(a.mean()), 'med': float(np.median(a)),
                                              'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90))}
        return {'n': len(uids), 'cer': float(100.0 * e.sum() / n.sum()), 'errors': float(e.sum()), 'chars': float(n.sum()),
                'lat': q(lat), 'real_L': q(rl), 'real_T': q(rt)}

    agg = {}
    for (s, m, c, L, T), d in per.items():
        if m == 'transformer' and not subset[s] <= set(d):
            continue  # some shards still decoding: report nothing rather than a partial sample
        uids = sorted(subset[s]) if m == 'transformer' else sorted(d)
        agg[f'{s}|{m}|{c}|{L}|{T}'] = summarize(d, uids)

    # paired bootstrap of condition differences on the shared segments (B = 1000)
    rng = np.random.default_rng(0)

    def boot(a, b, uids):
        uids = [u for u in uids if u in a and u in b]
        ea = np.array([a[u][0] for u in uids], float); eb = np.array([b[u][0] for u in uids], float)
        n = np.array([a[u][1] for u in uids], float)
        idx = rng.integers(0, len(uids), (1000, len(uids)))
        d = 100.0 * (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
        return {'diff': float(100.0 * (eb.sum() - ea.sum()) / n.sum()), 'lo': float(np.percentile(d, 2.5)), 'hi': float(np.percentile(d, 97.5))}

    opt = {s: json.load(open(EXP / 'opt' / f"{SETS[s].get('dev', s)}.json")) for s in SETS
           if (EXP / 'opt' / f"{SETS[s].get('dev', s)}.json").exists()}
    cmp_ = {}
    for s in SETS:
        if s not in opt:
            continue
        L, T = opt[s]['L'], opt[s]['T']
        for m in ('wav2vec2', 'transformer', 'conformer'):
            keys = {'base': (s, m, 'base', None, None), 'fa00': (s, m, 'fa', 0, 0), 'faopt': (s, m, 'fa', L, T),
                    'sil00': (s, m, 'sil', 0, 0), 'silopt': (s, m, 'sil', L, T)}
            if not all(k in per for k in keys.values()):
                continue
            if m == 'transformer' and not all(subset[s] <= set(per[k]) for k in keys.values()):
                continue
            uids = sorted(subset[s]) if m == 'transformer' else sorted(per[keys['fa00']])
            for a, b in [('base', 'faopt'), ('fa00', 'faopt'), ('sil00', 'silopt'), ('base', 'silopt'), ('faopt', 'silopt'), ('base', 'sil00')]:
                cmp_[f'{s}|{m}|{a}->{b}'] = boot(per[keys[a]], per[keys[b]], uids)
    # Segment-level Silero (AMI eval): padding effect, and against Base / FA / whole-recording Silero
    for s in ('ami_eval_ihm', 'ami_eval_sdm'):
        if s not in opt:
            continue
        L, T = opt[s]['L'], opt[s]['T']
        for m in ('wav2vec2', 'transformer', 'conformer'):
            keys = {'base': (s, m, 'base', None, None), 'faopt': (s, m, 'fa', L, T), 'sil00': (s, m, 'sil', 0, 0),
                    'silopt': (s, m, 'sil', L, T), 'ssil00': (s, m, 'ssil', 0, 0), 'ssilopt': (s, m, 'ssil', L, T)}
            if not all(k in per for k in keys.values()):
                continue
            uids = sorted(subset[s]) if m == 'transformer' else sorted(per[keys['base']])
            if m == 'transformer' and not all(subset[s] <= set(per[k]) for k in keys.values()):
                continue
            for a, b in [('ssil00', 'ssilopt'), ('base', 'ssilopt'), ('faopt', 'ssilopt'), ('silopt', 'ssilopt'), ('sil00', 'ssil00')]:
                cmp_[f'{s}|{m}|{a}->{b}'] = boot(per[keys[a]], per[keys[b]], uids)
    # Cutting into the speech (wav2vec2): CER increase from FA 0/0 when the first / last 50 or 100 ms of
    # the FA span are removed, and the paired difference "leading cut - trailing cut"
    cut = {}
    for s in SETS:
        ref = per.get((s, 'wav2vec2', 'fa', 0, 0))
        for mag in (50, 100):
            a, b = per.get((s, 'wav2vec2', 'fa', -mag, 0)), per.get((s, 'wav2vec2', 'fa', 0, -mag))
            if ref and a and b:
                uids = sorted(set(ref) & set(a) & set(b))
                cut[f'{s}|{mag}'] = {'cer00': 100.0 * sum(ref[u][0] for u in uids) / sum(ref[u][1] for u in uids),
                                     'lead': boot(ref, a, uids), 'trail': boot(ref, b, uids), 'lead_minus_trail': boot(b, a, uids)}
    # LibriSpeech: re-pick the dev optimum on segments with >= 500 ms of audio on both sides of the FA span
    from scripts.mv2.common import SR, GRID
    head = {}
    for s in ('ls_dev_clean', 'ls_dev_other'):
        fr = fa_rows(s)
        full = {r['uid'] for r in fr if r['fa_start'] - r['seg_start'] >= 0.499 and r['rec_len'] / SR - r['fa_end'] >= 0.499}
        g = {}
        for L, T in GRID:
            d = per.get((s, 'wav2vec2', 'fa', L, T))
            if d:
                us = [u for u in full if u in d]
                g[(L, T)] = 100.0 * sum(d[u][0] for u in us) / max(1, sum(d[u][1] for u in us))
        if len(g) == len(GRID) and s in opt:
            b = min(g, key=lambda c: (round(g[c], 6), c[0] + c[1]))
            o = (opt[s]['L'], opt[s]['T'])
            head[s] = {'n': len(full), 'opt_all': f'{o[0]}/{o[1]}', 'opt_sub': f'{b[0]}/{b[1]}', 'cer_sub_00': g[(0, 0)],
                       'cer_sub_opt': g[b], 'cer_sub_at_all_opt': g[o], 'grid': {f'{L}_{T}': v for (L, T), v in g.items()}}
    write_json(RES / 'ls_headroom.json', head)
    write_json(RES / 'agg.json', {'agg': agg, 'cmp': cmp_, 'cut': cut, 'opt': opt, 'scored_files': len(files), 'score_s': time.time() - t0})
    print(f'STAGE5 scored {len(files)} files, {len(agg)} cells in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
