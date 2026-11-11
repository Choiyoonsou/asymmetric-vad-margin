#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vad-margin-gen
export OMP_NUM_THREADS=4

# Decomposition control for the headroom claim. --cap-vad-only gives the detector
# the same short clip as the cap condition, but lets the policy pad into the full
# clip. If the sign flip survives, it is the detector's segmentation; if it
# vanishes, it is the padding room.
CAP=350
CELLS="0:0 100:0 200:0 300:0 400:0 500:0 400:400 400:100"

for SPLIT in dev test ihm sdm; do
  for CELL in $CELLS; do
    L="${CELL%%:*}"; T="${CELL##*:}"
    echo "=== $SPLIT  l=$L t=$T  cap=$CAP vad-only ==="
    python scripts/run_vad_decode.py \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      --vad silero --leading-pad-ms "$L" --trailing-pad-ms "$T" \
      --context-cap-ms "$CAP" --cap-vad-only \
      --exp-subdir cap_decomp \
      --model wav2vec2-base-960h --language none
  done
done
echo "[done] cap_decomp complete"
