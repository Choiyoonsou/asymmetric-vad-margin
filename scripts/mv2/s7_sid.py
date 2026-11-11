"""Stage 7: split CER into substitutions / deletions / insertions, and locate word errors at the utterance edges.

Eval sets, three models, five conditions per set: Base, FA 0/0, FA + dev optimum, Silero 0/0, Silero + dev optimum
(Silero = segment-level VAD 'ssil' on AMI eval, whole-recording 'sil' elsewhere, as in the report's Part 2).
Reads exp/mv2/results/segments.csv.gz (normalized ref / hyp), aligns with Levenshtein edit operations.

  char level   S, D, I in % of reference characters (S + D + I = corpus CER)
  word level   S, D, I in % of reference words, and edge errors: share of segments whose first / last reference
               word is deleted or substituted, and words inserted before the first / after the last reference word

Writes exp/mv2/results/sid.json.

    python scripts/mv2/s7_sid.py
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapidfuzz.distance import Levenshtein

from scripts.mv2.common import EVAL_SETS, EXP, write_json

RES = EXP / 'results'
FIELDS = ['nc', 'cs', 'cd', 'ci', 'nw', 'ws', 'wd', 'wi', 'n', 'first_d', 'first_s', 'last_d', 'last_s', 'ins_head', 'ins_tail', 'empty']


def wanted():
    """(set, model, cond, L, T) -> condition label, for the cells the report shows."""
    A = json.load(open(RES / 'agg.json'))
    agg, opt = A['agg'], A['opt']
    out = {}
    for s in EVAL_SETS:
        o = opt[s]
        L, T = str(o['L']), str(o['T'])
        sil = 'ssil' if f'{s}|wav2vec2|ssil|0|0' in agg else 'sil'
        for m in ('wav2vec2', 'transformer', 'conformer'):
            out[(s, m, 'base', '', '')] = 'Base'
            out[(s, m, 'fa', '0', '0')] = 'FA 0/0'
            out[(s, m, 'fa', L, T)] = 'FA+opt' if (L, T) != ('0', '0') else 'FA 0/0'
            out[(s, m, sil, '0', '0')] = 'Sil 0/0'
            out[(s, m, sil, L, T)] = 'Sil+opt' if (L, T) != ('0', '0') else 'Sil 0/0'
    return out


def edge_ops(rw, hw):
    """Per reference word: 'ok' / 'S' / 'D'; plus inserted-word counts before the first and after the last ref word."""
    tag = ['ok'] * len(rw)
    ins_pos = []
    for op in Levenshtein.editops(rw, hw):
        if op.tag == 'replace':
            tag[op.src_pos] = 'S'
        elif op.tag == 'delete':
            tag[op.src_pos] = 'D'
        else:
            ins_pos.append(op.src_pos)  # inserted before ref word src_pos
    head = sum(1 for p in ins_pos if p == 0)
    tail = sum(1 for p in ins_pos if p == len(rw))
    return tag, head, tail


def run(chunk):
    acc = defaultdict(lambda: [0] * len(FIELDS))
    for key, ref, hyp in chunk:
        a = acc[key]
        c = {'replace': 0, 'delete': 0, 'insert': 0}
        for op in Levenshtein.editops(ref, hyp):
            c[op.tag] += 1
        rw, hw = ref.split(), hyp.split()
        w = {'replace': 0, 'delete': 0, 'insert': 0}
        for op in Levenshtein.editops(rw, hw):
            w[op.tag] += 1
        tag, head, tail = edge_ops(rw, hw)
        vals = [len(ref), c['replace'], c['delete'], c['insert'], len(rw), w['replace'], w['delete'], w['insert'], 1,
                tag[0] == 'D', tag[0] == 'S', tag[-1] == 'D', tag[-1] == 'S', head, tail, not hyp]
        for i, v in enumerate(vals):
            a[i] += v
    return dict(acc)


def main():
    want = wanted()
    subset = {s: set(json.load(open(EXP / 'queue' / 'subset' / f'{s}.json'))) for s in EVAL_SETS}
    seen, rows = set(), []
    with gzip.open(RES / 'segments.csv.gz', 'rt') as fh:
        for r in csv.DictReader(fh):
            k = (r['set'], r['model'], r['cond'], r['L_req'], r['T_req'])
            if k not in want or not r['ref']:
                continue
            if r['model'] == 'transformer' and r['uid'] not in subset[r['set']]:
                continue
            if (k, r['uid']) in seen:  # SDM: the 0/0 file and the "optimum" file are the same cell
                continue
            seen.add((k, r['uid']))
            rows.append(((r['set'], r['model'], want[k]), r['ref'], r['hyp']))
    chunks = [rows[i::96] for i in range(96)]
    tot = defaultdict(lambda: [0] * len(FIELDS))
    with ProcessPoolExecutor(48) as ex:
        for part in ex.map(run, chunks):
            for k, v in part.items():
                t = tot[k]
                for i, x in enumerate(v):
                    t[i] += x
    out = {'|'.join(k): dict(zip(FIELDS, v)) for k, v in tot.items()}
    write_json(RES / 'sid.json', out)
    print(f'{len(rows)} rows, {len(out)} cells', flush=True)


if __name__ == '__main__':
    main()
