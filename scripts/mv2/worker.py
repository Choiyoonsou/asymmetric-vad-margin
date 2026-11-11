"""GPU worker: pulls decode / select jobs from exp/mv2/queue/jobs.json until none are left.

A job is claimed by creating its lock file (O_EXCL); a lock whose process is gone is taken over. A job is
done once its output exists (written atomically), so a killed run resumes where it stopped. Jobs whose
dependencies (grid shards, a dev-set optimum) are not ready yet are skipped and retried later.

  torch env (vad-margin-gen): wav2vec2 (batch 1: the base model's group norm sees padding, so batching
                              would change the output), Transformer (padded batches with relative
                              lengths, masked), and the CPU 'select' jobs
  nemo env (vmg-nemo):        FastConformer hybrid, CTC head, look-ahead 1,040 ms

    CUDA_VISIBLE_DEVICES=0 python scripts/mv2/worker.py --env torch --prefer wav2vec2 --name g0a
"""
import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.chdir(Path(__file__).resolve().parents[2])

import numpy as np

from scripts.mv2.common import EXP, GRID, SR, base_rows, fa_rows, jl, load_opt, opt_path, write_jl, write_json

Q = EXP / 'queue'
ENV_MODELS = {'torch': {'wav2vec2', 'transformer', 'select'}, 'nemo': {'conformer'}}
_models = {}


def out_path(job):
    return opt_path(job['set'], job.get('opt_tag', '')) if job['kind'] == 'select' else EXP / 'dec' / f"{job['id']}.jsonl"


def done(job):
    return out_path(job).exists()


def deps_met(job):
    for d in job.get('deps', []):
        if d.startswith('opt:'):
            if not opt_path(d[4:]).exists():
                return False
        elif not (EXP / 'dec' / f'{d}.jsonl').exists():
            return False
    return True


def claim(job):
    lock = Q / 'locks' / f"{job['id']}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({'pid': os.getpid(), 'host': socket.gethostname(), 't': time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                info = json.load(open(lock))
            except (ValueError, OSError):
                # created but not written yet by a worker claiming it right now: only a lock that stays
                # unreadable for a minute is treated as abandoned
                if time.time() - lock.stat().st_mtime < 60:
                    return False
                lock.unlink(missing_ok=True)
                continue
            try:
                os.kill(info['pid'], 0)
                return False  # owner alive
            except ProcessLookupError:
                lock.unlink(missing_ok=True)  # owner gone: take over
            except PermissionError:
                return False
    return False


# ---------------------------------------------------------------- models
def model(name):
    if name in _models:
        return _models[name]
    import torch
    if name == 'wav2vec2':
        import torchaudio
        b = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        m = b.get_model().cuda().eval()
        _models[name] = (m, b.get_labels())
    elif name == 'transformer':
        from speechbrain.inference.ASR import EncoderDecoderASR
        hf = 'speechbrain/asr-transformer-transformerlm-librispeech'
        _models[name] = EncoderDecoderASR.from_hparams(source=hf, savedir=f"artifacts/speechbrain/{hf.replace('/', '__')}",
                                                       run_opts={'device': 'cuda:0'})
    elif name == 'conformer':
        import logging
        for noisy in ('nemo_logger', 'nemo', 'lightning', 'pytorch_lightning'):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained('nvidia/stt_en_fastconformer_hybrid_large_streaming_multi').cuda().eval()
        m.change_decoding_strategy(decoder_type='ctc', verbose=False)
        m.encoder.set_default_att_context_size([70, 13])  # look-ahead 1,040 ms
        _models[name] = m
    return _models[name]


def decode_w2v(items, max_batch_samples=16 * SR * 20):
    """Batch only windows of identical length (no padding, so the output equals batch-1 decoding), and
    decode byte-identical windows once (LibriSpeech margins clipped at the file edge repeat the same audio)."""
    import hashlib
    import torch
    from src.asr import decode_tokens
    m, labels = model('wav2vec2')
    out = [''] * len(items)
    uniq = {}
    for i, a in enumerate(items):
        if a.size >= 400:  # shorter than one conv receptive field: nothing to recognize
            uniq.setdefault((a.size, hashlib.blake2b(a.tobytes(), digest_size=16).digest()), []).append(i)
    by_len = {}
    for (n, _), idx in uniq.items():
        by_len.setdefault(n, []).append(idx)
    for n, groups in by_len.items():
        step = max(1, max_batch_samples // n)
        for k in range(0, len(groups), step):
            chunk = groups[k:k + step]
            x = torch.from_numpy(np.stack([items[g[0]] for g in chunk])).cuda()
            with torch.inference_mode():
                em, _ = m(x)
            ids = torch.argmax(em, dim=-1).cpu().tolist()
            for g, t in zip(chunk, ids):
                h = decode_tokens(t, labels)
                for i in g:
                    out[i] = h
    return out


def decode_tr(items, max_batch_sec=96.0, max_items=32):
    import torch
    m = model('transformer')
    order = sorted(range(len(items)), key=lambda i: items[i].size)
    out = [''] * len(items)

    def run(idx):
        sizes = [items[i].size for i in idx]
        mx = max(sizes)
        batch = torch.zeros(len(idx), mx)
        for k, i in enumerate(idx):
            batch[k, :sizes[k]] = torch.from_numpy(items[i])
        rel = torch.tensor([s / mx for s in sizes])
        try:
            with torch.inference_mode():
                hyps, _ = m.transcribe_batch(batch.cuda(), rel.cuda())
            for i, h in zip(idx, hyps):
                out[i] = (h or '').strip().lower()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                # a single very long input (Silero merges neighbouring TED utterances): decode it on the CPU
                from speechbrain.inference.ASR import EncoderDecoderASR
                if 'transformer_cpu' not in _models:
                    hf = 'speechbrain/asr-transformer-transformerlm-librispeech'
                    _models['transformer_cpu'] = EncoderDecoderASR.from_hparams(
                        source=hf, savedir=f"artifacts/speechbrain/{hf.replace('/', '__')}", run_opts={'device': 'cpu'})
                with torch.inference_mode():
                    hyps, _ = _models['transformer_cpu'].transcribe_batch(batch, rel)
                out[idx[0]] = (hyps[0] or '').strip().lower()
                print(f'  CPU fallback for one {sizes[0] / SR:.1f}s input', flush=True)
                return
            run(idx[:len(idx) // 2])
            run(idx[len(idx) // 2:])

    batch = []
    for i in order:
        if items[i].size < 1600:  # < 0.1 s
            continue
        if batch and (len(batch) >= max_items or (len(batch) + 1) * items[i].size / SR > max_batch_sec):
            run(batch)
            batch = []
        batch.append(i)
    if batch:
        run(batch)
    return out


def decode_fc(items):
    import torch
    m = model('conformer')
    out = [''] * len(items)
    idx = [i for i, a in enumerate(items) if a.size >= 1600]
    for k in range(0, len(idx), 512):
        chunk = idx[k:k + 512]
        with torch.inference_mode():
            hyps = m.transcribe([items[i] for i in chunk], batch_size=32, verbose=False)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        for i, h in zip(chunk, hyps):
            out[i] = (h.text if hasattr(h, 'text') else h).strip().lower()
    return out


DECODERS = {'wav2vec2': decode_w2v, 'transformer': decode_tr, 'conformer': decode_fc}


# ---------------------------------------------------------------- jobs
def job_rows(job):
    rows = base_rows(job['set']) if job['cond'] == 'base' else fa_rows(job['set'])
    if job.get('subset'):
        keep = set(json.load(open(EXP / 'queue' / 'subset' / f"{job['set']}.json")))
        rows = [r for r in rows if r['uid'] in keep]
    i, n = job['shard']
    return rows[i::n]


def job_cells(job):
    if job['cond'] == 'base':
        return [(None, None)]
    if job['cells'] == 'opt':
        return [load_opt(job['set'], job.get('opt_tag', ''))]
    if job['cells'] == 'grid':
        return GRID
    return [tuple(c) for c in job['cells']]


def run_decode(job):
    from scripts.mv2.conditions import build
    rows, cells = job_rows(job), job_cells(job)
    pairs = [(r, c) for r in rows for c in cells]
    t0 = time.time()
    # build the audio on a CPU thread while the GPU decodes the previous block
    step = len(cells) * 3 if len(cells) > 1 else 256  # grid: whole utterances per block, so equal-length windows batch together
    blocks = [pairs[k:k + step] for k in range(0, len(pairs), step)]
    q = queue.Queue(maxsize=3)

    def producer():
        for blk in blocks:
            built = []
            for r, (L, T) in blk:
                a, info = build(r, job['cond'], L or 0, T or 0)
                built.append((r, L, T, a, info))
            q.put(built)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    out, audio_s, dec_s = [], 0.0, 0.0
    while True:
        built = q.get()
        if built is None:
            break
        td = time.time()
        hyps = DECODERS[job['model']]([b[3] for b in built])
        dec_s += time.time() - td
        for (r, L, T, a, info), h in zip(built, hyps):
            audio_s += a.size / SR
            out.append({'uid': r['uid'], 'set': job['set'], 'model': job['model'], 'cond': job['cond'], 'L': L, 'T': T,
                        'hyp': h, 'dec_sec': round(a.size / SR, 3), **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items()}})
    write_jl(out_path(job), out)
    write_json(EXP / 'timing' / f"{job['id']}.json", {'id': job['id'], 'model': job['model'], 'set': job['set'], 'cond': job['cond'],
                                                    'wall_s': time.time() - t0, 'decode_s': dec_s, 'audio_s': audio_s, 'n': len(out),
                                                    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})


def run_select(job):
    """Dev-set optimum: lowest corpus CER over a 121-cell wav2vec2 grid (FA, or segment-level Silero with opt_tag
    'ssil_'); ties -> smaller L + T."""
    import editdistance
    from scripts.summarize_metaseg import norm
    refs = {r['uid']: norm(r['transcript']) for r in fa_rows(job['set'])}
    e, n = {}, {}
    for d in job['deps']:
        for x in jl(EXP / 'dec' / f'{d}.jsonl'):
            c = (x['L'], x['T'])
            r = refs[x['uid']]
            e[c] = e.get(c, 0) + editdistance.eval(r, norm(x['hyp']))
            n[c] = n.get(c, 0) + len(r)
    cer = {c: 100.0 * e[c] / n[c] for c in e}
    best = min(cer, key=lambda c: (round(cer[c], 6), c[0] + c[1]))
    tag = job.get('opt_tag', '')
    write_json(EXP / 'grid' / f"{tag}{job['set']}.json", {'set': job['set'], 'cells': {f'{L}_{T}': v for (L, T), v in cer.items()}})
    write_json(opt_path(job['set'], tag), {'set': job['set'], 'cond': job.get('grid_cond', 'fa'), 'L': best[0], 'T': best[1], 'cer': cer[best], 'cer_00': cer[(0, 0)],
                                      'n_cells': len(cer)})
    print(f"[select] {job['set']}: optimum {best[0]}/{best[1]} CER {cer[best]:.3f} (0/0 {cer[(0, 0)]:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--env', choices=list(ENV_MODELS), required=True)
    ap.add_argument('--prefer', default='')
    ap.add_argument('--name', default=socket.gethostname())
    ap.add_argument('--jobs', default=str(Q / 'jobs.json'))
    ap.add_argument('--only', default='', help='comma list of job ids (benchmarking)')
    ap.add_argument('--models', default='', help='comma list: take only these models\' jobs (e.g. wav2vec2,select)')
    args = ap.parse_args()
    prefer = [p for p in args.prefer.split(',') if p]
    jobs = [j for j in json.load(open(args.jobs)) if j['model'] in ENV_MODELS[args.env]]
    if args.models:
        allowed = set(args.models.split(','))
        jobs = [j for j in jobs if j['model'] in allowed]
    if args.only:
        keep = set(args.only.split(','))
        jobs = [j for j in jobs if j['id'] in keep]
    rank = lambda j: (j['model'] not in prefer if prefer else False, j['prio'], -j['est'])
    jobs.sort(key=rank)
    print(f"[{args.name}] start: {len(jobs)} {args.env} jobs, gpu {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    while True:
        pending = [j for j in jobs if not done(j)]
        if not pending:
            print(f'[{args.name}] no jobs left, exiting', flush=True)
            return
        ran = False
        for j in pending:
            if not deps_met(j) or not claim(j):
                continue
            if done(j):
                continue
            t0 = time.time()
            print(f"[{args.name}] run {j['id']} (est {j['est'] / 3600:.2f} h audio)", flush=True)
            try:
                run_select(j) if j['kind'] == 'select' else run_decode(j)
                print(f"[{args.name}] done {j['id']} in {time.time() - t0:.0f}s", flush=True)
            except Exception:
                print(f"[{args.name}] FAILED {j['id']}\n{traceback.format_exc()}", flush=True)
                (Q / 'locks' / f"{j['id']}.lock").unlink(missing_ok=True)
                (Q / 'failed').mkdir(parents=True, exist_ok=True)
                (Q / 'failed' / f"{j['id']}.txt").write_text(traceback.format_exc())
                time.sleep(5)
            ran = True
            break
        if not ran:
            time.sleep(20)


if __name__ == '__main__':
    main()
