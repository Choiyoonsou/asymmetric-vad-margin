#!/bin/bash
# Phase B: starts once every wav2vec2 / select job has its output (checked on the files, not a log line).
# Per GPU: one Transformer worker and one Conformer worker (the Conformer queue is short; the
# Transformer's padded beam-search batches need most of a T4's memory).
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
N=${PY_NEMO:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
w2v_left() {
  $P -c "
import json, sys; sys.path.insert(0, '.')
from scripts.mv2.worker import done
print(sum(1 for j in json.load(open('exp/mv2/queue/jobs.json')) if j['model'] in ('wav2vec2', 'select') and not done(j)))"
}
until [ "$(w2v_left 2>/dev/null)" = "0" ]; do sleep 60; done
echo "phase B start $(date)"
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g nohup $P scripts/mv2/worker.py --env torch --models transformer --name g${g}tr > logs/mv2/worker_tr_g$g.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g nohup $N scripts/mv2/worker.py --env nemo --models conformer --name g${g}fc > logs/mv2/worker_fc_g$g.log 2>&1 &
done
nohup $P scripts/mv2/monitor.py --models transformer,conformer --every 60 > /dev/null 2>&1 &
wait
echo "PHASE_B_DONE $(date)"
