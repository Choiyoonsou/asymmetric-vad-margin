"""Audio fed to the recognizer under each condition, with requested / realized margins and STT-input latency.

  base      the metadata segment [seg_start, seg_end] as is
  fa        FA span [fa_start - L, fa_end + T], clipped to the recording. LibriSpeech recordings are the
            distributed utterance files, so large margins are cut at the file edge; the realized margin is
            recorded next to the requested one.
  sil       Silero segments (speech_pad_ms = 0) of the whole recording that overlap the FA span, each
            extended by L / T (clipped, overlaps merged) and concatenated. (L, T) = (0, 0) is "no padding".
  ssil      as sil, but Silero runs on the base window +- 0.5 s only (AMI; scripts/mv2/s3b_vad_seg.py), and the
            padding is clipped to that window, so no other speaker's turn outside it reaches the recognizer.

STT-input latency (ms): time from the end of the last FA word until the segment can be handed to the
recognizer, decode time excluded.
  base  seg_end - fa_end                      (the metadata segment's implicit trailing margin)
  fa    realized trailing margin              (requested T is kept in lat_req_ms)
  sil   max(e + min_silence + lookahead, e + T) - fa_end, e = raw end of the last matched Silero segment
  ssil  as sil, with e + T replaced by the realized (window-clipped) padded end
"""
import json
from functools import lru_cache

import numpy as np

from scripts.mv2.common import SILERO, SILERO_LOOKAHEAD_MS, SR, load_rec, window
from scripts.mv2.s3_vad import vad_path

ENDPOINT_S = (SILERO['min_silence_duration_ms'] + SILERO_LOOKAHEAD_MS) / 1000.0


@lru_cache(maxsize=4096)
def vad_segments(rec_path):
    return np.array(json.load(open(vad_path(rec_path)))['segments'], dtype=np.int64).reshape(-1, 2)


def build(row, cond, L=0, T=0):
    """(audio float32, info dict) for one base row under one condition."""
    rec = load_rec(row['rec_path'])
    n = rec.shape[0]
    fa_s, fa_e = row.get('fa_start'), row.get('fa_end')
    if cond == 'base':
        a, b = int(round(row['seg_start'] * SR)), min(n, int(round(row['seg_end'] * SR)))
        if fa_e is None:  # no alignment: latency and realized margins are filled in at scoring
            return np.asarray(rec[a:b], np.float32), {'real_L': None, 'real_T': None, 'lat_ms': None, 'lat_req_ms': None}
        lat = 1000.0 * (row['seg_end'] - fa_e)
        return np.asarray(rec[a:b], np.float32), {'real_L': 1000.0 * (fa_s - row['seg_start']), 'real_T': lat,
                                                  'lat_ms': lat, 'lat_req_ms': lat}
    if cond == 'fa':
        a, b, rl, rt = window(n, fa_s, fa_e, L, T)
        return np.asarray(rec[a:b], np.float32), {'real_L': rl, 'real_T': rt, 'lat_ms': rt, 'lat_req_ms': float(T)}
    if cond == 'sil':
        segs = vad_segments(row['rec_path'])
        s0, e0 = int(round(fa_s * SR)), int(round(fa_e * SR))
        hit = segs[(segs[:, 1] > s0) & (segs[:, 0] < e0)] if len(segs) else segs
        if not len(hit):
            return np.zeros(0, np.float32), {'real_L': None, 'real_T': None, 'lat_ms': None, 'lat_req_ms': None, 'n_vad': 0}
        padded = np.stack([np.maximum(0, hit[:, 0] - int(round(L * SR / 1000))), np.minimum(n, hit[:, 1] + int(round(T * SR / 1000)))], 1)
        merged = [list(padded[0])]
        for s, e in padded[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        audio = np.concatenate([np.asarray(rec[s:e], np.float32) for s, e in merged])
        e_raw = hit[-1, 1] / SR
        ready = max(e_raw + ENDPOINT_S, e_raw + T / 1000.0)
        return audio, {'real_L': 1000.0 * (fa_s - merged[0][0] / SR), 'real_T': 1000.0 * (merged[-1][1] / SR - fa_e),
                       'lat_ms': 1000.0 * (ready - fa_e), 'lat_req_ms': 1000.0 * (ready - fa_e), 'n_vad': len(hit)}
    if cond == 'ssil':  # Silero run on the base window +- 0.5 s only (scripts/mv2/s3b_vad_seg.py)
        v = vadseg(row['set'])[row['uid']]
        wa, wb = v['win']
        segs = np.array(v['segments'], dtype=np.int64).reshape(-1, 2)
        s0, e0 = int(round(fa_s * SR)), int(round(fa_e * SR))
        hit = segs[(segs[:, 1] > s0) & (segs[:, 0] < e0)] if len(segs) else segs
        if not len(hit):
            return np.zeros(0, np.float32), {'real_L': None, 'real_T': None, 'lat_ms': None, 'lat_req_ms': None, 'n_vad': 0}
        padded = np.stack([np.maximum(wa, hit[:, 0] - int(round(L * SR / 1000))), np.minimum(wb, hit[:, 1] + int(round(T * SR / 1000)))], 1)
        merged = [list(padded[0])]
        for s, e in padded[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        audio = np.concatenate([np.asarray(rec[s:e], np.float32) for s, e in merged])
        e_raw = hit[-1, 1] / SR
        ready = max(e_raw + ENDPOINT_S, merged[-1][1] / SR)
        return audio, {'real_L': 1000.0 * (fa_s - merged[0][0] / SR), 'real_T': 1000.0 * (merged[-1][1] / SR - fa_e),
                       'lat_ms': 1000.0 * (ready - fa_e), 'lat_req_ms': 1000.0 * (max(e_raw + ENDPOINT_S, e_raw + T / 1000.0) - fa_e),
                       'n_vad': len(hit)}
    raise ValueError(cond)


@lru_cache(maxsize=8)
def vadseg(s):
    from scripts.mv2.s3b_vad_seg import vadseg_path
    return json.load(open(vadseg_path(s)))
