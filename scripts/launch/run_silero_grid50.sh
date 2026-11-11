#!/bin/bash
# Fill the Silero margin grid (exp/margin_grid) out to a 50 ms step on both axes.
# Only the 73 (leading, trailing) cells missing from the original 48-cell grid are run.
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vad-margin-gen
export OMP_NUM_THREADS=4
SPLIT="$1"
for L in 0 50 100 150 200 250 300 350 400 450 500; do
  for T in 0 50 100 150 200 250 300 350 400 450 500; do
    OUT=$(printf "exp/margin_grid/decodes/%s__silero_vad_grid_l%03d_t%03d__wav2vec2_base_960h.jsonl" "$SPLIT" "$L" "$T")
    [[ -s "$OUT" ]] && continue
    echo "=== $SPLIT l=$L t=$T $(date +%T) ==="
    python scripts/run_vad_decode.py \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      --vad silero --leading-pad-ms "$L" --trailing-pad-ms "$T" \
      --exp-subdir margin_grid \
      --model wav2vec2-base-960h --language none || echo "FAILED $SPLIT $L $T"
  done
done
echo "SILERO_GRID50_DONE $SPLIT $(date)"
