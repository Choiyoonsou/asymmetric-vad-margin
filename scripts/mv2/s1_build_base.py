"""Stage 1: base segments from the raw corpora, the 30 s filter, and the audio cache.

  LibriSpeech  each distributed utterance file is one base segment (start 0, end = file length)
  TED-LIUM 3   .stm segments of the raw .sph talks (inter_segment_gap / ignore_time_segment_in_scoring rows
               are not speech segments and are not listed)
  AMI          segments.xml transcriber start/end on the raw recordings; IHM = the speaker's headset,
               SDM = Array1-01 (same segments, same ids)

Only filter: base segments of 30 s or longer are dropped (recorded per set). Segments that cannot be
scored at all -- empty transcript or end <= start -- are listed separately as invalid.

    python scripts/mv2/s1_build_base.py --workers 32
    -> artifacts/mv2/base/<set>.jsonl, artifacts/mv2/base/filter_stats.json, artifacts/mv2/audio_cache/*.npy
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

from scripts.mv2.common import ART, CACHE, MAX_BASE_SEC, ROOT, SETS, SR, cache_path, write_jl, write_json
from src.data.ami import collect_segment_words, load_speaker_channels, normalize_transcript, parse_segments, parse_words, source_audio_path
from src.data.tedlium import parse_stm

DATA = ROOT / 'data'
AMI_MEETINGS = {
    'dev': ['ES2011a', 'ES2011b', 'ES2011c', 'ES2011d', 'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d',
            'TS3004a', 'TS3004b', 'TS3004c', 'TS3004d', 'IB4001', 'IB4002', 'IB4003', 'IB4004', 'IB4010', 'IB4011'],
    'eval': ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d', 'ES2004a', 'ES2004b', 'ES2004c', 'ES2004d',
             'IS1009a', 'IS1009b', 'IS1009c', 'IS1009d', 'TS3003a', 'TS3003b', 'TS3003c', 'TS3003d'],
}


def to_cache(rec_path):
    """Decode one recording to 16 kHz mono float32 .npy (skipped when already cached)."""
    out = cache_path(rec_path)
    if out.exists():
        return str(rec_path), int(np.load(out, mmap_mode='r').shape[0])
    if str(rec_path).endswith('.sph'):
        from src.audio.io import read_audio
        audio, sr = read_audio(rec_path)
    else:
        audio, sr = sf.read(rec_path, dtype='float32', always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SR).numpy()
    tmp = out.with_name(out.name + '.tmp.npy')
    np.save(tmp, audio.astype(np.float32))
    tmp.rename(out)
    return str(rec_path), int(audio.shape[0])


def ls_rows(split):
    rows = []
    for trans in sorted((DATA / 'LibriSpeech' / split).glob('*/*/*.trans.txt')):
        for line in trans.read_text().splitlines():
            uid, text = line.split(' ', 1)
            path = trans.parent / f'{uid}.flac'
            rows.append({'uid': uid, 'rec_path': str(path), 'speaker': uid.split('-')[0], 'transcript': text,
                         'seg_start': 0.0, 'seg_end': None})  # end = file length, filled after caching
    return rows


def ted_rows(split):
    rows = []
    for stm in sorted((DATA / 'TEDLIUM_raw' / split).glob('*.stm')):
        for i, seg in enumerate(parse_stm(stm)):
            rows.append({'uid': f"{seg['talk_id']}_{i:04d}", 'rec_path': str(stm.with_suffix('.sph')), 'speaker': seg['talk_id'],
                         'transcript': seg['transcript'], 'seg_start': float(seg['start']), 'seg_end': float(seg['end'])})
    return rows


def ami_rows(split, mic):
    root = DATA / 'AMI_raw'
    ann = root / 'annotations' / 'unzipped'
    channels = load_speaker_channels(ann)
    rows = []
    for meeting in AMI_MEETINGS[split]:
        for agent, channel in sorted(channels.get(meeting, {}).items()):
            words_path = ann / 'words' / f'{meeting}.{agent}.words.xml'
            seg_path = ann / 'segments' / f'{meeting}.{agent}.segments.xml'
            src = source_audio_path(root / 'audio', meeting, mic, channel)
            if not (words_path.exists() and seg_path.exists() and src.exists()):
                print(f'  skip {meeting}.{agent} ({mic}): missing annotation or audio', flush=True)
                continue
            words = parse_words(words_path)
            index = {w['id']: k for k, w in enumerate(words) if w['id']}
            for i, seg in enumerate(parse_segments(seg_path)):
                seg_words = collect_segment_words(words, index, seg['start_id'], seg['end_id'])
                text = normalize_transcript(seg_words) if seg_words else ''
                rows.append({'uid': f'{meeting}_{agent}_{i:04d}', 'rec_path': str(src), 'speaker': f'{meeting}_{agent}',
                             'transcript': text, 'seg_start': float(seg['start']), 'seg_end': float(seg['end']),
                             'ihm_rec_path': str(source_audio_path(root / 'audio', meeting, 'ihm', channel))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    args = ap.parse_args()
    t0 = time.time()
    raw = {}
    for s in args.sets:
        c = SETS[s]
        raw[s] = ls_rows(c['split']) if c['corpus'] == 'ls' else ted_rows(c['split']) if c['corpus'] == 'ted' else ami_rows(c['split'], c['mic'])
        print(f'{s}: {len(raw[s])} metadata segments', flush=True)

    recs = sorted({r['rec_path'] for rows in raw.values() for r in rows} | {r['ihm_rec_path'] for rows in raw.values() for r in rows if 'ihm_rec_path' in r})
    CACHE.mkdir(parents=True, exist_ok=True)
    lengths = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (p, n) in enumerate(ex.map(to_cache, recs, chunksize=8)):
            lengths[p] = n
            if (k + 1) % 2000 == 0 or k + 1 == len(recs):
                print(f'  cached {k + 1}/{len(recs)} recordings ({time.time() - t0:.0f}s)', flush=True)

    stats_path = ART / 'base' / 'filter_stats.json'
    stats = {}
    if stats_path.exists():
        import json
        stats = json.load(open(stats_path))
    for s, rows in raw.items():
        kept, invalid, long_ = [], 0, 0
        for r in rows:
            r['set'] = s
            r['rec_len'] = lengths[r['rec_path']]
            if r['seg_end'] is None:
                r['seg_end'] = r['rec_len'] / SR
            r['seg_end'] = min(r['seg_end'], r['rec_len'] / SR)
            dur = r['seg_end'] - r['seg_start']
            if not r['transcript'].strip() or dur <= 0:
                invalid += 1
                continue
            if dur >= MAX_BASE_SEC:
                long_ += 1
                continue
            r['dur'] = round(dur, 4)
            kept.append(r)
        write_jl(ART / 'base' / f'{s}.jsonl', kept)
        valid = len(rows) - invalid
        stats[s] = {'metadata_segments': len(rows), 'invalid': invalid, 'valid': valid, 'removed_30s': long_,
                    'removed_30s_pct': 100.0 * long_ / max(valid, 1), 'kept': len(kept),
                    'kept_hours': sum(r['dur'] for r in kept) / 3600}
        print(f"{s}: valid {valid}, >=30 s removed {long_} ({stats[s]['removed_30s_pct']:.2f}%), kept {len(kept)} "
              f"({stats[s]['kept_hours']:.2f} h), invalid {invalid}", flush=True)
    write_json(stats_path, stats)
    print(f'STAGE1 done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
