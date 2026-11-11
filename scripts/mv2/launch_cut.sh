#!/bin/bash
# Cut-margin jobs (wav2vec2, FA with -50/-100 ms at the start or the end), on the GPUs not used by the last Transformer job.
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in 0 2 3 4 5; do
  for k in a b; do
    CUDA_VISIBLE_DEVICES=$g nohup $P scripts/mv2/worker.py --env torch --models wav2vec2 --name cut_g$g$k > logs/mv2/worker_cut_g$g$k.log 2>&1 &
  done
done
wait
echo "CUT_DONE $(date)"
