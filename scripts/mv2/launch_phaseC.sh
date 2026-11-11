#!/bin/bash
# Phase C: once every job in the queue has its output, score everything.
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
export PYTHONWARNINGS=ignore
left() {
  $P -c "
import json, sys; sys.path.insert(0, '.')
from scripts.mv2.worker import done
print(sum(1 for j in json.load(open('exp/mv2/queue/jobs.json')) if not done(j)))"
}
until [ "$(left 2>/dev/null)" = "0" ]; do sleep 120; done
echo "phase C start $(date)"
$P scripts/mv2/s5_score.py > logs/mv2/s5_score.log 2>&1 && echo "scored $(date)"
echo "PHASE_C_DONE $(date)"
