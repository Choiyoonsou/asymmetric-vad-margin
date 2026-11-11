#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vad-margin-gen
export OMP_NUM_THREADS=4

# Headroom control. AMI/TED-LIUM clips carry 1000ms of context on each side;
# LibriSpeech carries a median of ~350/330ms. Capping at 350ms reproduces
# LibriSpeech's condition on datasets that otherwise have room, so the margin
# surface can be compared with and without the confound on the SAME utterances.
CAP=350
# The padding axes, where saturation shows, plus the two policy cells.
CELLS="0:0 100:0 200:0 300:0 400:0 500:0 0:100 0:200 0:300 0:400 0:500 400:400 400:100"

for SPLIT in dev test ihm sdm; do
  for CELL in $CELLS; do
    L="${CELL%%:*}"; T="${CELL##*:}"
    echo "=== $SPLIT  l=$L t=$T  cap=$CAP ==="
    python scripts/run_vad_decode.py \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      --vad silero --leading-pad-ms "$L" --trailing-pad-ms "$T" \
      --context-cap-ms "$CAP" \
      --exp-subdir cap_control \
      --model wav2vec2-base-960h --language none
  done
done
echo "[done] cap_control complete"
