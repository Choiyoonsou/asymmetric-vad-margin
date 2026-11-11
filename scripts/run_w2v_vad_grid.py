"""Fast wav2vec2 decode of a Silero margin grid from saved VAD segments.

scripts/run_vad_decode.py re-runs Silero for every margin cell. Here the raw segments
saved by scripts/export_vad_segments.py are reused: each cell only applies the margin
policy (apply_vad_policy), concatenates the segments, and decodes with wav2vec2. The
audio fed to the recognizer is the same as run_vad_decode builds. Cells whose output
already exists are skipped, so this can share a grid with run_vad_decode loops.

    python scripts/run_w2v_vad_grid.py --split dev-clean --step 100 --reverse
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchaudio

from src.asr import AsrSettings, Wav2Vec2Runner, decode_tokens
from src.audio.io import concatenate_segments, read_audio
from src.vad_policy import VadPolicySettings, apply_vad_policy


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--split', required=True, help='Name used in exp/vad_segments/<split>__silero.jsonl and the output files.')
    p.add_argument('--step', type=int, default=100)
    p.add_argument('--reverse', action='store_true', help='Walk the grid from the largest margins down.')
    p.add_argument('--cells', nargs='+', default=None, help='Explicit L,T cells instead of the grid, e.g. 450,300.')
    args = p.parse_args()
    seg_path = PROJECT_ROOT / 'exp' / 'vad_segments' / f'{args.split}__silero.jsonl'
    records = [json.loads(l) for l in open(seg_path)]
    out_dir = PROJECT_ROOT / 'exp' / 'margin_grid' / 'decodes'
    cells = [(L, T) for L in range(0, 501, args.step) for T in range(0, 501, args.step)]
    if args.cells:
        cells = [tuple(int(v) for v in c.split(',')) for c in args.cells]
    if args.reverse:
        cells = cells[::-1]
    runner = Wav2Vec2Runner(AsrSettings(model_name='wav2vec2-base-960h', device='cuda', device_index=0))
    audio_cache = {}

    def audio_of(rec):
        if rec['utterance_id'] not in audio_cache:
            a, sr = read_audio(rec['audio_path'])
            if a.ndim > 1:
                a = a.mean(axis=-1)
            if sr != rec['sample_rate']:
                a = torchaudio.functional.resample(torch.from_numpy(a), sr, rec['sample_rate']).numpy()
            audio_cache[rec['utterance_id']] = a.astype(np.float32)[: rec['num_samples']]
        return audio_cache[rec['utterance_id']]

    for L, T in cells:
        out = out_dir / f'{args.split}__silero_vad_grid_l{L:03d}_t{T:03d}__wav2vec2_base_960h.jsonl'
        if out.exists() and out.stat().st_size > 0:
            continue
        policy = VadPolicySettings(policy_name=f'grid_l{L:03d}_t{T:03d}', leading_pad_ms=L, trailing_pad_ms=T)
        rows = []
        for rec in records:
            a = audio_of(rec)
            segs = [{'start': s, 'end': e} for s, e in rec['segments']]
            if L or T:
                segs = apply_vad_policy(segments=segs, sample_rate=rec['sample_rate'], audio_length=len(a), settings=policy)
            pieces = [a[s['start']:s['end']] for s in segs]
            clip = np.asarray(concatenate_segments(pieces), np.float32) if pieces else np.zeros(0, np.float32)
            hyp = ''
            if clip.size:
                with torch.inference_mode():
                    em, _ = runner.model(torch.from_numpy(clip).unsqueeze(0).to(runner.device))
                hyp = decode_tokens(torch.argmax(em, dim=-1)[0].cpu().tolist(), runner.labels)
            rows.append({'source_utterance_id': rec['utterance_id'], 'condition': f'silero_vad_grid_l{L:03d}_t{T:03d}',
                         'split': args.split, 'hypothesis': hyp, 'reference': rec.get('reference'),
                         'decoded_duration_sec': round(clip.size / rec['sample_rate'], 3),
                         'vad_policy_leading_pad_ms': L, 'vad_policy_trailing_pad_ms': T})
        if out.exists() and out.stat().st_size > 0:  # written meanwhile by another loop
            continue
        tmp = out.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        tmp.rename(out)
        print(f'{args.split} {L}/{T} done', flush=True)
    print(f'DONE {args.split}', flush=True)


if __name__ == '__main__':
    main()
