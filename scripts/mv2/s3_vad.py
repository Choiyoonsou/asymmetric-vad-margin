"""Stage 3: Silero VAD over every whole recording (speech_pad_ms = 0), cached per recording.

Silero runs on the full stream a deployed system would see -- the whole TED talk, the whole AMI headset
or array channel, the LibriSpeech file -- not on a clip around each segment. Raw speech segments (sample
indices, no padding) go to artifacts/mv2/vad/<cache key>.json. How they are matched to each base segment
and padded is decided at decode time (scripts/mv2/conditions.py).

    python scripts/mv2/s3_vad.py --workers 48
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import ART, SETS, SILERO, SR, base_rows, cache_path, load_rec, write_json

VAD_DIR = ART / 'vad'
_model = None


def vad_path(rec_path):
    return VAD_DIR / (cache_path(rec_path).stem + '.json')


def run_one(rec_path):
    global _model
    out = vad_path(rec_path)
    if out.exists():
        return rec_path, 0.0, 0.0
    import torch
    torch.set_num_threads(1)
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _model is None:
        _model = load_silero_vad()
    audio = torch.from_numpy(load_rec(rec_path)[:].copy())
    t0 = time.time()
    segs = get_speech_timestamps(audio, _model, sampling_rate=SR, threshold=SILERO['threshold'],
                                 min_silence_duration_ms=SILERO['min_silence_duration_ms'],
                                 speech_pad_ms=SILERO['speech_pad_ms'], max_speech_duration_s=SILERO['max_speech_duration_s'])
    write_json(out, {'rec_path': rec_path, 'settings': SILERO, 'segments': [[int(s['start']), int(s['end'])] for s in segs]})
    return rec_path, time.time() - t0, len(audio) / SR


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    recs = sorted({r['rec_path'] for s in SETS for r in base_rows(s)})
    # longest first so the big AMI channels do not end up last on one worker
    recs.sort(key=lambda p: -load_rec(p).shape[0])
    t0, cpu, audio_s = time.time(), 0.0, 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (p, dt, dur) in enumerate(ex.map(run_one, recs, chunksize=1)):
            cpu += dt
            audio_s += dur
            if (k + 1) % 500 == 0 or k + 1 == len(recs) or (k + 1) <= 100 and (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f'VAD {k + 1}/{len(recs)} recordings, {el:.0f}s elapsed, ~{el / (k + 1) * (len(recs) - k - 1):.0f}s left', flush=True)
    write_json(ART / 'vad' / '_timing.json', {'wall_s': time.time() - t0, 'cpu_s': cpu, 'audio_s': audio_s, 'recordings': len(recs)})
    print(f'STAGE3 VAD done: {len(recs)} recordings in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
