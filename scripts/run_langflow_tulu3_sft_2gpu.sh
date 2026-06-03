#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
LANGFLOW_REPO="${LANGFLOW_REPO:-../LangFlow}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-../checkpoints/Continuous-Rivals-Discrete/langflow-owt}"
DATA_PATH="${DATA_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/mmdata/text_sft/tulu3_sft_mixture_t5_1024_english}"
OUT_DIR="${OUT_DIR:-outputs/langflow-owt-tulu3-english-sft}"
RESUME="${RESUME:-none}"
TOKENIZER="${TOKENIZER:-gpt2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-768}"
SELF_COND_PROB="${SELF_COND_PROB:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

IFS=',' read -r -a device_list <<< "${CUDA_DEVICES}"
nproc="${#device_list[@]}"

local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_args+=(--local_files_only)
fi

common_args=(
  scripts/train_langflow_sft.py
  --langflow_repo "${LANGFLOW_REPO}"
  --init_checkpoint "${INIT_CHECKPOINT}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUT_DIR}"
  --resume "${RESUME}"
  --tokenizer "${TOKENIZER}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --lr "${LR}"
  --max_length "${MAX_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --self_cond_prob "${SELF_COND_PROB}"
  --num_workers "${NUM_WORKERS}"
  --save_every_steps "${SAVE_EVERY_STEPS}"
  --save_every_epoch
  --log_every "${LOG_EVERY}"
  "${local_args[@]}"
)

if [[ "${nproc}" -gt 1 ]]; then
  python3 -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "${common_args[@]}"
else
  python3 "${common_args[@]}"
fi
