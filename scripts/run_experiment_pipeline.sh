#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  echo "Usage: bash scripts/run_experiment_pipeline.sh <step1|step2|all_steps> <dataset> [limit] [--model NAME] [--language LANG] [--splits A,B]"
  echo
  echo "  --model     ASR backbone for this run (see src/asr.py:BACKBONE_REGISTRY)."
  echo "              Defaults to conf/base.yaml asr.model_name."
  echo "  --language  Decoding language; pass 'none' to clear it for CTC backbones."
  echo "  --splits    Comma-separated subset of the dataset's splits to run."
  echo "              Defaults to every split listed under the dataset in conf/base.yaml."
  echo "  --grid      Margin conditions: axes (baseline 1D sweeps, default),"
  echo "              grid2d (leading x trailing product), or both."
  echo "  --materialize  wav (one file per condition, default) or offsets"
  echo "              (record the window and slice at decode time -- no disk cost)."
  echo
  echo "Outputs are tagged <dataset>__<asr_tag>, so decoding the same dataset with"
  echo "several backbones no longer overwrites the previous backbone's eval CSVs."
}

read_pipeline_config() {
  python -m src.pipeline.read_pipeline_config --config "${CONFIG_PATH}" "$@"
}

run_step1() {
  echo "=== START step1 fa_exp | dataset=${DATASET} limit=${LIMIT:-all} ==="

  for SPLIT in "${SPLITS[@]}"; do
    echo "=== Running step1 fa_exp for dataset=${DATASET} split=${SPLIT} limit=${LIMIT:-all} ==="

    python scripts/build_manifest.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --dataset "${DATASET}" \
      --splits "${SPLIT}" \
      "${LIMIT_ARGS[@]}"

    python scripts/generate_oracle_variants.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --manifest "${MANIFEST_DIR}/${SPLIT}.jsonl" \
      --grid "${GRID}" \
      --materialize "${MATERIALIZE}" \
      --output-suffix "${VARIANT_SUFFIX}" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_asr_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --manifest "${MANIFEST_DIR}/${SPLIT}.jsonl" \
      "${ASR_ARGS[@]+"${ASR_ARGS[@]}"}" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_asr_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --manifest "${MANIFEST_DIR}/${SPLIT}_${VARIANT_SUFFIX}.jsonl" \
      "${ASR_ARGS[@]+"${ASR_ARGS[@]}"}"

    python scripts/summarize_results.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --inputs \
      "${EXP_PATH}/step1/decodes/${SPLIT}__${ASR_TAG}.jsonl" \
      "${EXP_PATH}/step1/decodes/${SPLIT}_${VARIANT_SUFFIX}__${ASR_TAG}.jsonl" \
      --output-prefix "${RUN_TAG}_${SPLIT}"
  done

  python -m src.analysis.render_step1_views \
    --config "${CONFIG_PATH}" \
    --logging-config "${LOGGING_CONFIG_PATH}" \
    --exp-subdir step1 \
    --dataset "${RUN_TAG}" \
    --output-dir "${EXP_PATH}/step1/analysis/${RUN_TAG}"

  echo "[done] step1 fa_exp completed successfully for dataset=${DATASET} limit=${LIMIT:-all}"
}

run_step2() {
  local policy_tag="silero_vad_${POLICY_NAME}"
  local step1_decode_dir="${EXP_PATH}/step1/decodes"
  local step2_decode_dir="${EXP_PATH}/step2/decodes"
  local manifest_dir="${MANIFEST_DIR}"

  echo "=== START step2 apply_vad | dataset=${DATASET} limit=${LIMIT:-all} ==="

  for SPLIT in "${SPLITS[@]}"; do
    echo "=== Running step2 apply_vad for dataset=${DATASET} split=${SPLIT} limit=${LIMIT:-all} ==="

    local manifest_path="${manifest_dir}/${SPLIT}.jsonl"
    local baseline_decode_path="${step1_decode_dir}/${SPLIT}__${ASR_TAG}.jsonl"

    if [[ ! -f "${manifest_path}" ]]; then
      echo "Missing manifest for step2: ${manifest_path}"
      echo "Run step1 first."
      exit 1
    fi

    if [[ ! -f "${baseline_decode_path}" ]]; then
      echo "Missing baseline decode for step2: ${baseline_decode_path}"
      echo "Run step1 first."
      exit 1
    fi

    local step2_baseline_input="${baseline_decode_path}"
    if [[ -n "${LIMIT}" ]]; then
      mkdir -p "${step2_decode_dir}"
      step2_baseline_input="${step2_decode_dir}/${SPLIT}__${ASR_TAG}__step2_limit${LIMIT}.jsonl"
      python -m src.eval.filter_decode_by_manifest \
        --manifest "${manifest_path}" \
        --input "${baseline_decode_path}" \
        --output "${step2_baseline_input}" \
        --limit "${LIMIT}"
    fi

    python scripts/run_silero_vad_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --policy default \
      --exp-subdir step2 \
      --manifest "${manifest_path}" \
      "${ASR_ARGS[@]+"${ASR_ARGS[@]}"}" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_silero_vad_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --policy fa_informed \
      --exp-subdir step2 \
      --manifest "${manifest_path}" \
      "${ASR_ARGS[@]+"${ASR_ARGS[@]}"}" \
      "${LIMIT_ARGS[@]}"

    python scripts/summarize_results.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step2 \
      --inputs \
      "${step2_baseline_input}" \
      "${EXP_PATH}/step2/decodes/${SPLIT}__silero_vad_default__${ASR_TAG}.jsonl" \
      "${EXP_PATH}/step2/decodes/${SPLIT}__${policy_tag}__${ASR_TAG}.jsonl" \
      --output-prefix "${RUN_TAG}_${SPLIT}"
  done

  python -m src.analysis.render_step2_views \
    --config "${CONFIG_PATH}" \
    --logging-config "${LOGGING_CONFIG_PATH}" \
    --exp-subdir step2 \
    --dataset "${RUN_TAG}" \
    --output-dir "${EXP_PATH}/step2/analysis/${RUN_TAG}"

  echo "[done] step2 apply_vad completed successfully for dataset=${DATASET} limit=${LIMIT:-all}"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

STAGE="${1}"
DATASET="${2}"
shift 2

LIMIT=""
MODEL=""
LANGUAGE=""
SPLITS_OVERRIDE=""
GRID="axes"
MATERIALIZE="wav"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  LIMIT="${1}"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --language)
      LANGUAGE="${2:-}"
      shift 2
      ;;
    --splits)
      SPLITS_OVERRIDE="${2:-}"
      shift 2
      ;;
    --grid)
      GRID="${2:-}"
      shift 2
      ;;
    --materialize)
      MATERIALIZE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: ${1}"
      usage
      exit 1
      ;;
  esac
done

case "${GRID}" in
  axes)   VARIANT_SUFFIX="oracle_variants" ;;
  grid2d) VARIANT_SUFFIX="margin_grid" ;;
  both)   VARIANT_SUFFIX="margin_full" ;;
  *) echo "Invalid --grid: ${GRID} (axes|grid2d|both)"; exit 1 ;;
esac

CONFIG_PATH="${PROJECT_ROOT}/conf/base.yaml"
LOGGING_CONFIG_PATH="${PROJECT_ROOT}/conf/logging.yaml"

if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
else
  LIMIT_ARGS=()
fi

ASR_ARGS=()
TAG_ARGS=()
if [[ -n "${MODEL}" ]]; then
  ASR_ARGS+=(--model "${MODEL}")
  TAG_ARGS+=(--model "${MODEL}")
fi
if [[ -n "${LANGUAGE}" ]]; then
  ASR_ARGS+=(--language "${LANGUAGE}")
fi

readarray -t DATASET_SPLITS < <(read_pipeline_config --dataset "${DATASET}" --field splits)

if [[ -n "${SPLITS_OVERRIDE}" ]]; then
  # A subset run still has to name splits the dataset actually declares --
  # a typo would otherwise decode nothing and then render an analysis
  # directory from whatever stale eval CSVs happened to share the run tag.
  IFS=',' read -r -a SPLITS <<< "${SPLITS_OVERRIDE}"
  for REQUESTED in "${SPLITS[@]}"; do
    FOUND=0
    for KNOWN in "${DATASET_SPLITS[@]}"; do
      [[ "${REQUESTED}" == "${KNOWN}" ]] && FOUND=1 && break
    done
    if [[ "${FOUND}" -eq 0 ]]; then
      echo "Unknown split '${REQUESTED}' for dataset '${DATASET}'. Known: ${DATASET_SPLITS[*]}"
      exit 1
    fi
  done
else
  SPLITS=("${DATASET_SPLITS[@]}")
fi
ASR_TAG="$(read_pipeline_config "${TAG_ARGS[@]+"${TAG_ARGS[@]}"}" --field asr_tag)"
POLICY_NAME="$(read_pipeline_config --field vad_policy_name)"
EXP_PATH="$(read_pipeline_config --field exp_path)"
MANIFEST_DIR="$(read_pipeline_config --field manifest_output_path)"

# Every eval CSV and analysis directory is keyed on dataset AND backbone. The
# render scripts glob "<run tag>_*_per_utterance.csv", so an untagged prefix
# would let a second backbone's run silently overwrite the first one's results.
RUN_TAG="${DATASET}__${ASR_TAG}"

case "${STAGE}" in
  step1)
    run_step1
    ;;
  step2)
    run_step2
    ;;
  all_steps)
    run_step1
    run_step2
    ;;
  *)
    echo "Invalid stage: ${STAGE}"
    usage
    exit 1
    ;;
esac
