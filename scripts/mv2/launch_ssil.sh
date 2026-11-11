#!/bin/bash
# Segment-level Silero ('ssil', AMI eval, FA dev optimum): per GPU one wav2vec2, one Transformer and one
# Conformer worker on exp/mv2/queue/jobs_ssil.json; when the list is done, rescore.
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
N=${PY_NEMO:-python}
J=exp/mv2/queue/jobs_ssil.json
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
left() {
  $P -c "
import json, sys; sys.path.insert(0, '.')
from scripts.mv2.worker import done
print(sum(1 for j in json.load(open('$J')) if not done(j)))"
}
echo "ssil start $(date)"
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g nohup $P scripts/mv2/worker.py --env torch --jobs $J --models wav2vec2 --name ss_g${g}w > logs/mv2/worker_ssil_w2v_g$g.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g nohup $P scripts/mv2/worker.py --env torch --jobs $J --models transformer --name ss_g${g}tr > logs/mv2/worker_ssil_tr_g$g.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g nohup $N scripts/mv2/worker.py --env nemo --jobs $J --models conformer --name ss_g${g}fc > logs/mv2/worker_ssil_fc_g$g.log 2>&1 &
done
until [ "$(left 2>/dev/null)" = "0" ]; do sleep 30; done
echo "ssil decoding done $(date)"
$P scripts/mv2/s5_score.py > logs/mv2/s5_score.log 2>&1 && echo "scored $(date)"
echo "SSIL_DONE $(date)"
