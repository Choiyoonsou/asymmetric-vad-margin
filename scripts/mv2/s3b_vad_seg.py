"""Stage 3b: Silero VAD per AMI base segment (segment-level VAD, condition 'ssil').

Stage 3 runs Silero on the whole recording, so on AMI (and the SDM array in particular) a VAD segment can run
through several speakers' turns and one base segment can receive seconds of other people's speech. Here Silero
runs on each base segment's own window [seg_start - CTX, seg_end + CTX] (clipped to the recording), with a fresh
model state per window, so what the STT sees stays inside that window. CTX = 0.5 s equals the largest margin in
the grid, so a requested padding can be realized beyond the metadata boundary.

Raw speech segments (absolute sample indices, no padding) per set: artifacts/mv2/vadseg/<set>.json
{uid: {"win": [a, b], "segments": [[s, e], ...]}}.

    python scripts/mv2/s3b_vad_seg.py --workers 48
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mv2.common import ART, SILERO, SR, fa_rows, load_rec, write_json

VADSEG_DIR = ART / 'vadseg'
CTX_S = 0.5
AMI_SETS = ['ami_dev_ihm', 'ami_dev_sdm', 'ami_eval_ihm', 'ami_eval_sdm']
_model = None


def vadseg_path(s):
    return VADSEG_DIR / f'{s}.json'


def seg_window(row, rec_len):
    return max(0, int(round((row['seg_start'] - CTX_S) * SR))), min(rec_len, int(round((row['seg_end'] + CTX_S) * SR)))


def run_rows(rows):
    global _model
    import torch
    torch.set_num_threads(1)
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _model is None:
        _model = load_silero_vad()
    out = {}
    t0 = time.time()
    for r in rows:
        rec = load_rec(r['rec_path'])
        a, b = seg_window(r, rec.shape[0])
        segs = get_speech_timestamps(torch.from_numpy(rec[a:b].copy()), _model, sampling_rate=SR, threshold=SILERO['threshold'],
                                     min_silence_duration_ms=SILERO['min_silence_duration_ms'],
                                     speech_pad_ms=SILERO['speech_pad_ms'], max_speech_duration_s=SILERO['max_speech_duration_s'])
        out[r['uid']] = {'win': [a, b], 'segments': [[a + int(s['start']), a + int(s['end'])] for s in segs]}
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--sets', default=','.join(AMI_SETS))
    args = ap.parse_args()
    timing = {}
    for s in args.sets.split(','):
        if vadseg_path(s).exists():
            print(f'{s}: exists, skipped', flush=True)
            continue
        rows = fa_rows(s)
        chunks = [rows[i::args.workers * 4] for i in range(args.workers * 4)]
        t0, cpu, res = time.time(), 0.0, {}
        with ProcessPoolExecutor(args.workers) as ex:
            for d, dt in ex.map(run_rows, chunks):
                res.update(d)
                cpu += dt
        write_json(vadseg_path(s), res)
        timing[s] = {'wall_s': time.time() - t0, 'cpu_s': cpu, 'n': len(res)}
        print(f'{s}: {len(res)} segments in {time.time() - t0:.0f}s', flush=True)
    if timing:
        write_json(VADSEG_DIR / '_timing.json', timing)


if __name__ == '__main__':
    main()
