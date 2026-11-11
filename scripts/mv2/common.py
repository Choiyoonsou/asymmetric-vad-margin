"""Shared definitions for the margin v2 experiment (raw corpora, 30 s filter, FA on base segments).

Sets
  dev   ls_dev_clean, ls_dev_other, ted_dev, ami_dev_ihm, ami_dev_sdm     (wav2vec2 121-cell grid here)
  eval  ls_test_clean, ls_test_other, ted_test, ami_eval_ihm, ami_eval_sdm (dev optimum applied unchanged)

Every recording is cached once as 16 kHz mono float32 .npy (artifacts/mv2/audio_cache) and read with
mmap, so margins, VAD and decoding all slice the same samples without re-decoding audio files.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'mv2'
EXP = ROOT / 'exp' / 'mv2'
CACHE = ART / 'audio_cache'
SR = 16000
MAX_BASE_SEC = 30.0

PY_TORCH = os.environ.get('PY_TORCH', 'python')  # interpreter of the main (torch) env
PY_NEMO = '$PY_NEMO'
MFA_BIN = os.environ.get('MFA_BIN', '')  # bin/ dir of the MFA env; empty = use PATH

SETS = {
    'ls_dev_clean': {'corpus': 'ls', 'split': 'dev-clean', 'role': 'dev', 'name': 'LS dev-clean'},
    'ls_dev_other': {'corpus': 'ls', 'split': 'dev-other', 'role': 'dev', 'name': 'LS dev-other'},
    'ted_dev': {'corpus': 'ted', 'split': 'dev', 'role': 'dev', 'name': 'TED-LIUM 3 dev'},
    'ami_dev_ihm': {'corpus': 'ami', 'split': 'dev', 'mic': 'ihm', 'role': 'dev', 'name': 'AMI dev IHM'},
    'ami_dev_sdm': {'corpus': 'ami', 'split': 'dev', 'mic': 'sdm', 'role': 'dev', 'name': 'AMI dev SDM'},
    'ls_test_clean': {'corpus': 'ls', 'split': 'test-clean', 'role': 'eval', 'dev': 'ls_dev_clean', 'name': 'LS test-clean'},
    'ls_test_other': {'corpus': 'ls', 'split': 'test-other', 'role': 'eval', 'dev': 'ls_dev_other', 'name': 'LS test-other'},
    'ted_test': {'corpus': 'ted', 'split': 'test', 'role': 'eval', 'dev': 'ted_dev', 'name': 'TED-LIUM 3 test'},
    'ami_eval_ihm': {'corpus': 'ami', 'split': 'eval', 'mic': 'ihm', 'role': 'eval', 'dev': 'ami_dev_ihm', 'name': 'AMI eval IHM'},
    'ami_eval_sdm': {'corpus': 'ami', 'split': 'eval', 'mic': 'sdm', 'role': 'eval', 'dev': 'ami_dev_sdm', 'name': 'AMI eval SDM'},
}
DEV_SETS = [s for s in SETS if SETS[s]['role'] == 'dev']
EVAL_SETS = [s for s in SETS if SETS[s]['role'] == 'eval']
MODELS = ['wav2vec2', 'transformer', 'conformer']
GRID = [(L, T) for L in range(0, 501, 50) for T in range(0, 501, 50)]

# Silero VAD: no padding; the remaining settings are the ones used throughout the project
SILERO = {'threshold': 0.5, 'min_silence_duration_ms': 100, 'speech_pad_ms': 0, 'max_speech_duration_s': 30.0}
SILERO_LOOKAHEAD_MS = 32.0  # Silero v5 reports a boundary only once its 512-sample window is complete


def dev_of(s):
    """The dev set whose optimum margin a set uses (itself for dev sets)."""
    return SETS[s].get('dev', s)


def jl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jl(path, rows):
    """Atomic JSONL write (tmp + rename) so a crashed run never leaves a half file that looks done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def cache_path(rec_path):
    h = hashlib.sha1(str(rec_path).encode()).hexdigest()[:16]
    return CACHE / f'{Path(rec_path).stem}.{h}.npy'


def load_rec(rec_path, _mem={}):
    """16 kHz mono float32 samples of a cached recording (memory-mapped, process-local memo)."""
    key = str(rec_path)
    if key not in _mem:
        _mem[key] = np.load(cache_path(rec_path), mmap_mode='r')
    return _mem[key]


def base_rows(s):
    return jl(ART / 'base' / f'{s}.jsonl')


def fa_rows(s):
    """FA spans of the evaluation set: base rows that passed the 30 s filter and have an alignment."""
    return jl(ART / 'fa' / f'{s}.jsonl')


def opt_path(dev_set, tag=''):
    """tag namespaces a second optimum for the same dev set (e.g. 'ssil_' for the segment-level Silero grid)."""
    return EXP / 'opt' / f'{tag}{dev_set}.json'


def load_opt(s, tag=''):
    p = opt_path(dev_of(s), tag)
    if not p.exists():
        return None
    d = json.load(open(p))
    return d['L'], d['T']


def window(rec_len, start, end, L, T):
    """[start - L, end + T] in samples, clipped to the recording; returns (a, b, realized_L_ms, realized_T_ms)."""
    a = max(0, int(round((start - L / 1000.0) * SR)))
    b = min(rec_len, int(round((end + T / 1000.0) * SR)))
    return a, b, 1000.0 * (start - a / SR), 1000.0 * (b / SR - end)
