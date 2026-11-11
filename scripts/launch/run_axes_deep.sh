#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vad-margin-gen
export OMP_NUM_THREADS=4
CFG=conf/axes_deep.yaml
for SPLIT in dev test ihm sdm test-clean test-other; do
  echo "=== $SPLIT : generating offsets ==="
  python scripts/generate_oracle_variants.py \
    --config "$CFG" --manifest "artifacts/manifests/${SPLIT}.jsonl" \
    --grid axes --materialize offsets --no-pause --output-suffix axes_deep
  echo "=== $SPLIT : decoding ==="
  python scripts/run_asr_decode.py \
    --config "$CFG" --exp-subdir axes_deep \
    --manifest "artifacts/manifests/${SPLIT}_axes_deep.jsonl" \
    --model wav2vec2-base-960h --language none
done
echo "[done] axes_deep complete"
