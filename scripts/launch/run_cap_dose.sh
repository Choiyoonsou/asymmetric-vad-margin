#!/bin/bash
# Headroom dose-response. run_cap_control.sh established that capping context to
# 350ms (LibriSpeech's median) flips the sign of the padding effect. One cap is a
# single point; the causal claim wants a curve. These caps, plus the 350ms run
# already on disk and the uncapped grid (1000ms of context), give five levels of
# available headroom on the same utterances.
#
# One split per GPU, pinned through conf/gpu_<split>.yaml, because
# set_cuda_visible_devices() takes CUDA_VISIBLE_DEVICES from the config.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vad-margin-gen
export OMP_NUM_THREADS=4

SPLIT="$1"
CAPS="150 250 500"
CELLS="0:0 100:0 200:0 300:0 400:0 500:0 0:100 0:200 0:300 0:400 0:500 400:400 400:100"

for CAP in $CAPS; do
  for CELL in $CELLS; do
    L="${CELL%%:*}"; T="${CELL##*:}"
    echo "=== $SPLIT  l=$L t=$T  cap=$CAP ==="
    python scripts/run_vad_decode.py \
      --config "conf/gpu_${SPLIT}.yaml" \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      --vad silero --leading-pad-ms "$L" --trailing-pad-ms "$T" \
      --context-cap-ms "$CAP" \
      --exp-subdir cap_dose \
      --model wav2vec2-base-960h --language none
  done
done
echo "[done] cap_dose $SPLIT complete"
