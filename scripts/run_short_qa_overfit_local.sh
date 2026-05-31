#!/usr/bin/env bash
# Run the short-QA high-noise overfit experiment directly on the dev server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-src/configs/training_configs/train_short_qa_overfit_ELF-L_local_2gpu.yml}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/models/t5-small}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/short_qa_overfit/t5_1024_eos}"
TRAIN_JSONL="${TRAIN_JSONL:-${REPO_ROOT}/data/short_qa_overfit/short_qa_overfit_train.jsonl}"
PROBE_JSONL="${PROBE_JSONL:-${REPO_ROOT}/eval_probes/short_qa_overfit_probe.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/short_qa_overfit_ELF-L_t5_1024_from_sft5402}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${REPO_ROOT}/outputs/elf_l-tulu3-sft-t5-1024-hope-32gpu/checkpoint_5402}"

NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-96}"
NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29631}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ELF_IMPORT_STAGGER_SEC="${ELF_IMPORT_STAGGER_SEC:-2}"

python scripts/prepare_short_qa_overfit.py \
  --tokenizer "${TOKENIZER}" \
  --output_dir "${DATA_DIR}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --probe_jsonl "${PROBE_JSONL}" \
  --num_examples "${NUM_EXAMPLES}" \
  --max_length "${MAX_LENGTH}" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --append_eos

export CUDA_VISIBLE_DEVICES
export MASTER_PORT
export ELF_IMPORT_STAGGER_SEC

echo "[short-qa-overfit] repo=${REPO_ROOT}"
echo "[short-qa-overfit] data=${DATA_DIR}"
echo "[short-qa-overfit] probe=${PROBE_JSONL}"
echo "[short-qa-overfit] output=${OUTPUT_DIR}"
echo "[short-qa-overfit] init=${INIT_CHECKPOINT}"
echo "[short-qa-overfit] cuda=${CUDA_VISIBLE_DEVICES} ngpu=${NGPU}"

NGPU="${NGPU}" bash scripts/launch.sh train "${CONFIG}" \
  --config_override "data_path=${DATA_DIR}" \
  --config_override "encoder_model_name=${TOKENIZER}" \
  --config_override "tokenizer_name=${TOKENIZER}" \
  --config_override "output_dir=${OUTPUT_DIR}" \
  --config_override "init_checkpoint=${INIT_CHECKPOINT}" \
  --config_override "max_length=${MAX_LENGTH}" \
  --config_override "max_input_length=${MAX_INPUT_LENGTH}" \
  "$@"
