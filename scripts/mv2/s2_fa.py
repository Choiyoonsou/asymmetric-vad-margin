"""Stage 2: forced alignment (MFA) of every base segment with its transcript.

  prepare  write one MFA corpus per corpus/split: <speaker>/<uid>.wav (+ .lab) holding exactly the base
           segment (LibriSpeech: a link to the distributed file). AMI is aligned on the IHM headset audio.
  align    run `mfa align` on one corpus (english_us_mfa dictionary, english_mfa acoustic model,
           beam 100 / retry beam 400 so fewer segments fail).
  collect  read the TextGrids: FA span = first word start .. last word end, shifted to recording time.
           Segments without an aligned word are FA failures and leave every condition (recorded per set).
           AMI SDM takes the span of the same segment's IHM alignment (the recordings are synchronized).

    python scripts/mv2/s2_fa.py prepare --workers 32
    python scripts/mv2/s2_fa.py align --group ls_dev -j 24     (mfa-aligner env, called via subprocess)
    python scripts/mv2/s2_fa.py collect
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import soundfile as sf

from scripts.mv2.common import ART, MFA_BIN, SETS, SR, base_rows, load_rec, write_jl, write_json

MFA_DIR = ART / 'mfa'
MODELS = Path.home() / 'Documents' / 'MFA' / 'pretrained_models'
# one MFA corpus per (corpus, split); AMI SDM shares the IHM alignment
GROUPS = {'ls_dev': ['ls_dev_clean', 'ls_dev_other'], 'ls_test': ['ls_test_clean', 'ls_test_other'],
          'ted_dev': ['ted_dev'], 'ted_test': ['ted_test'], 'ami_dev': ['ami_dev_ihm'], 'ami_eval': ['ami_eval_ihm']}


def mfa_text(text):
    """Transcript as MFA sees it: lower case, tags dropped, TED's split contractions ("it 's") rejoined."""
    t = re.sub(r'<[^>]*>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)', ' ', text.lower())
    t = re.sub(r"\s+'(?=[a-z])", "'", t)
    t = re.sub(r"[^a-z' \-]", ' ', t).replace('-', ' ')
    return ' '.join(w for w in t.split() if w.strip("'"))


def write_item(args):
    row, corpus = args
    d = corpus / row['speaker']
    d.mkdir(parents=True, exist_ok=True)
    text = mfa_text(row['transcript'])
    if not text:
        return 0
    wav = d / f"{row['uid']}.wav"
    if row['set'].startswith('ls_'):
        dst = d / f"{row['uid']}.flac"
        if not dst.exists():
            os.symlink(row['rec_path'], dst)
    elif not wav.exists():
        a = load_rec(row['rec_path'])
        sf.write(wav, a[int(round(row['seg_start'] * SR)):int(round(row['seg_end'] * SR))], SR)
    (d / f"{row['uid']}.lab").write_text(text + '\n')
    return 1


def prepare(workers):
    for g, sets in GROUPS.items():
        corpus = MFA_DIR / g / 'corpus'
        rows = [r for s in sets for r in base_rows(s)]
        with ProcessPoolExecutor(workers) as ex:
            n = sum(ex.map(write_item, [(r, corpus) for r in rows], chunksize=64))
        print(f'{g}: {n}/{len(rows)} segments written to the MFA corpus', flush=True)


def align(group, jobs):
    g = MFA_DIR / group
    (g / 'beam.yaml').write_text('beam: 100\nretry_beam: 400\n')
    path = f"{MFA_BIN}:{os.environ['PATH']}" if MFA_BIN else os.environ['PATH']
    env = dict(os.environ, PATH=path, MFA_ROOT_DIR=str(g / 'mfa_root'))
    cmd = ['mfa', 'align', '--clean', '-j', str(jobs), '--output_format', 'long_textgrid', '-c', str(g / 'beam.yaml'),
           '--temporary_directory', str(g / 'tmp'), str(g / 'corpus'), str(MODELS / 'dictionary' / 'english_us_mfa.dict'),
           str(MODELS / 'acoustic' / 'english_mfa.zip'), str(g / 'out')]
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    print(f'MFA {group} exit {r.returncode} in {time.time() - t0:.0f}s', flush=True)
    return r.returncode


def words_of(tg_path):
    from textgrid import TextGrid
    grid = TextGrid.fromFile(str(tg_path))
    out = []
    for tier in grid.tiers:
        if 'word' in tier.name.lower():
            out += [(float(iv.minTime), float(iv.maxTime)) for iv in tier.intervals if (iv.mark or '').strip()]
    return sorted(out)


def collect():
    stats = {}
    ihm_span = {}
    for g, sets in GROUPS.items():
        out_dir = MFA_DIR / g / 'out'
        for s in sets:
            kept, failed = [], []
            for r in base_rows(s):
                tg = out_dir / r['speaker'] / f"{r['uid']}.TextGrid"
                w = words_of(tg) if tg.exists() else []
                if not w:
                    failed.append(r['uid'])
                    continue
                r['fa_start'] = r['seg_start'] + w[0][0]
                r['fa_end'] = r['seg_start'] + w[-1][1]
                r['n_fa_words'] = len(w)
                kept.append(r)
                ihm_span[r['uid']] = (r['fa_start'], r['fa_end'], len(w))
            write_jl(ART / 'fa' / f'{s}.jsonl', kept)
            stats[s] = {'base': len(kept) + len(failed), 'fa_failed': len(failed), 'kept': len(kept),
                        'fa_failed_pct': 100.0 * len(failed) / max(1, len(kept) + len(failed)), 'failed_uids': failed}
            print(f"{s}: FA ok {len(kept)}, failed {len(failed)} ({stats[s]['fa_failed_pct']:.2f}%)", flush=True)
    for s in ('ami_dev_sdm', 'ami_eval_sdm'):
        kept, failed = [], []
        for r in base_rows(s):
            if r['uid'] not in ihm_span:
                failed.append(r['uid'])
                continue
            r['fa_start'], r['fa_end'], r['n_fa_words'] = ihm_span[r['uid']]
            kept.append(r)
        write_jl(ART / 'fa' / f'{s}.jsonl', kept)
        stats[s] = {'base': len(kept) + len(failed), 'fa_failed': len(failed), 'kept': len(kept),
                    'fa_failed_pct': 100.0 * len(failed) / max(1, len(kept) + len(failed)), 'failed_uids': failed,
                    'note': 'FA span taken from the IHM alignment of the same segment'}
        print(f"{s}: FA ok {len(kept)}, failed {len(failed)} (IHM alignment)", flush=True)
    write_json(ART / 'fa' / 'fa_stats.json', stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['prepare', 'align', 'collect'])
    ap.add_argument('--group', choices=list(GROUPS))
    ap.add_argument('-j', type=int, default=16)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    if args.stage == 'prepare':
        prepare(args.workers)
    elif args.stage == 'align':
        sys.exit(align(args.group, args.j))
    else:
        collect()


if __name__ == '__main__':
    main()
