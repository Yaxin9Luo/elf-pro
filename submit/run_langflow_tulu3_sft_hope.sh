#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro"
LANGFLOW_REPO="${LANGFLOW_REPO:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/LangFlow}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/checkpoints/Continuous-Rivals-Discrete/langflow-owt}"
DATA_PATH="${DATA_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/mmdata/text_sft/tulu3_sft_mixture_t5_1024_english}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/langflow-owt-tulu3-english-sft-20ep-32gpu}"
RESUME="${RESUME:-auto}"
CONDA_ROOT="/usr/local/conda"
ELF_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin"

source "${CONDA_ROOT}/bin/activate" "${ELF_ENV}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${LANGFLOW_REPO}:${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_DIST_INIT_TIMEOUT_MIN=10
export NCCL_DEBUG=WARN

mkdir -p "${OUT_DIR}/launch_logs"
LAUNCH_LOG="${OUT_DIR}/launch_logs/worker_${AFO_TASK_ID:-unknown}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "===== LangFlow Tulu3 SFT HOPE launch $(date '+%F %T') ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "langflow_repo=${LANGFLOW_REPO}"
echo "init_checkpoint=${INIT_CHECKPOINT}"
echo "data_path=${DATA_PATH}"
echo "out_dir=${OUT_DIR}"
echo "resume=${RESUME}"
echo "conda_env=${CONDA_PREFIX:-}"
echo "AFO_TASK_ID=${AFO_TASK_ID:-}"
echo "AFO_ENV_CLUSTER_SPEC=${AFO_ENV_CLUSTER_SPEC:-}"

TORCHRUN_FLAGS="$(python3 submit/hope_run_torch_distribute.py)"
echo "torchrun_flags=${TORCHRUN_FLAGS}"

HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run ${TORCHRUN_FLAGS} \
  scripts/train_langflow_sft.py \
  --langflow_repo "${LANGFLOW_REPO}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUT_DIR}" \
  --resume "${RESUME}" \
  --tokenizer gpt2 \
  --epochs "${EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --lr "${LR:-2e-5}" \
  --max_length "${MAX_LENGTH:-1024}" \
  --max_target_length "${MAX_TARGET_LENGTH:-768}" \
  --self_cond_prob "${SELF_COND_PROB:-0.5}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --save_every_steps "${SAVE_EVERY_STEPS:-0}" \
  --save_every_epoch \
  --log_every "${LOG_EVERY:-20}" \
  --local_files_only
