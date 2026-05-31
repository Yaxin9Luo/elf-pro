#!/usr/bin/env bash
# Evaluate a short-QA overfit checkpoint with clean decode, controlled denoise,
# and target-start trajectory probes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-src/configs/training_configs/train_short_qa_overfit_ELF-L_local_2gpu.yml}"
PROBE_JSONL="${PROBE_JSONL:-${REPO_ROOT}/eval_probes/short_qa_overfit_probe.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/short_qa_overfit_ELF-L_t5_1024_from_sft5402}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/probes}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-96}"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls -t "${OUTPUT_DIR}"/checkpoint_* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Pass one explicitly or train into ${OUTPUT_DIR}." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[short-qa-probe] checkpoint=${CHECKPOINT}"
echo "[short-qa-probe] result_dir=${RESULT_DIR}"

python eval_probes/run_cfm_sanity_checks.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompts "${PROBE_JSONL}" \
  --output "${RESULT_DIR}/cfm_sanity_short_qa_overfit_params.json" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --batch_size 8 \
  --timesteps "0.9,0.7,0.5,0.3,0.1" \
  --use_params

python eval_probes/run_sampling_trajectory_probe.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompts "${PROBE_JSONL}" \
  --output "${RESULT_DIR}/sampling_trajectory_short_qa_overfit_params.json" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --max_examples 16 \
  --seed 123 \
  --t_starts "0.95,0.7,0.5,0.3,0.1,0.0" \
  --uniform_steps 32 \
  --logit_steps 64 \
  --trace_every 8 \
  --use_params

echo "[short-qa-probe] wrote ${RESULT_DIR}"
