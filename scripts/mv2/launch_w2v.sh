#!/bin/bash
# Phase A: every wav2vec2 job (dev grid -> optima -> base / Silero / FA on dev and eval), 2 workers per GPU.
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
for g in 0 1 2 3 4 5; do
  for k in a b; do
    CUDA_VISIBLE_DEVICES=$g nohup $P scripts/mv2/worker.py --env torch --models wav2vec2,select --name g$g$k > logs/mv2/worker_w2v_g$g$k.log 2>&1 &
  done
done
nohup $P scripts/mv2/monitor.py --models wav2vec2 --every 60 > /dev/null 2>&1 &
wait
echo PHASE_A_DONE
