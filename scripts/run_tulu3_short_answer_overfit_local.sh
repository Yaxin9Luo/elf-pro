#!/usr/bin/env bash
# Run a real Tulu3 short-answer overfit experiment directly on the dev server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-src/configs/training_configs/train_short_qa_overfit_ELF-L_local_2gpu.yml}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/models/t5-small}"
DATA_DIR="${DATA_DIR:-/tmp/elf-pro-cache/data/tulu3_short_answer_clean_en_10k_t5_1024_eos}"
PROBE_JSONL="${PROBE_JSONL:-/tmp/elf-pro-cache/data/tulu3_short_answer_clean_en_10k_probe.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/elf-pro-cache/tulu3_short_answer_clean_en_10k_ELF-L_t5_1024_from_sft5402}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/tmp/elf-pro-cache/checkpoint_5402}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-256}"
EPOCHS="${EPOCHS:-20}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-999999}"
LOG_FREQ="${LOG_FREQ:-20}"
NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29645}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ELF_IMPORT_STAGGER_SEC="${ELF_IMPORT_STAGGER_SEC:-2}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export MASTER_PORT
export ELF_IMPORT_STAGGER_SEC

echo "[tulu3-short-answer] repo=${REPO_ROOT}"
echo "[tulu3-short-answer] data=${DATA_DIR}"
echo "[tulu3-short-answer] probe=${PROBE_JSONL}"
echo "[tulu3-short-answer] output=${OUTPUT_DIR}"
echo "[tulu3-short-answer] init=${INIT_CHECKPOINT}"
echo "[tulu3-short-answer] cuda=${CUDA_VISIBLE_DEVICES} ngpu=${NGPU}"
echo "[tulu3-short-answer] gbs=${GLOBAL_BATCH_SIZE} epochs=${EPOCHS} max_input=${MAX_INPUT_LENGTH}"

NGPU="${NGPU}" bash scripts/launch.sh train "${CONFIG}" \
  --config_override "data_path=${DATA_DIR}" \
  --config_override "encoder_model_name=${TOKENIZER}" \
  --config_override "tokenizer_name=${TOKENIZER}" \
  --config_override "output_dir=${OUTPUT_DIR}" \
  --config_override "init_checkpoint=${INIT_CHECKPOINT}" \
  --config_override "max_length=${MAX_LENGTH}" \
  --config_override "max_input_length=${MAX_INPUT_LENGTH}" \
  --config_override "epochs=${EPOCHS}" \
  --config_override "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  --config_override "save_freq=${SAVE_FREQ}" \
  --config_override "max_checkpoints_to_keep=1" \
  --config_override "log_freq=${LOG_FREQ}" \
  --config_override "wandb_run_name=$(basename "${OUTPUT_DIR}")" \
  "$@"
