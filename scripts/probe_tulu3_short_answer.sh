#!/usr/bin/env bash
# Evaluate a Tulu3 short-answer overfit checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-src/configs/training_configs/train_short_qa_overfit_ELF-L_local_2gpu.yml}"
PROBE_JSONL="${PROBE_JSONL:-/tmp/elf-pro-cache/data/tulu3_short_answer_clean_en_10k_probe.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/elf-pro-cache/tulu3_short_answer_clean_en_10k_ELF-L_t5_1024_from_sft5402}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/probes}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-256}"
MAX_EXAMPLES="${MAX_EXAMPLES:-64}"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls -t "${OUTPUT_DIR}"/checkpoint_* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Pass one explicitly or train into ${OUTPUT_DIR}." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[tulu3-short-answer-probe] checkpoint=${CHECKPOINT}"
echo "[tulu3-short-answer-probe] prompts=${PROBE_JSONL}"
echo "[tulu3-short-answer-probe] result_dir=${RESULT_DIR}"

python eval_probes/run_cfm_sanity_checks.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompts "${PROBE_JSONL}" \
  --output "${RESULT_DIR}/cfm_sanity_tulu3_short_answer_params.json" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --batch_size 8 \
  --timesteps "0.9,0.7,0.5,0.3,0.1" \
  --use_params

python eval_probes/run_sampling_trajectory_probe.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --prompts "${PROBE_JSONL}" \
  --output "${RESULT_DIR}/sampling_trajectory_tulu3_short_answer_params.json" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed 123 \
  --t_starts "0.95,0.7,0.5,0.3,0.1,0.0" \
  --uniform_steps 32 \
  --logit_steps 64 \
  --trace_every 8 \
  --use_params

echo "[tulu3-short-answer-probe] wrote ${RESULT_DIR}"
