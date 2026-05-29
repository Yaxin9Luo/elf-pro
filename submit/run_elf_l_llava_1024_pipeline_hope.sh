#!/usr/bin/env bash
# HOPE worker.script for the full 1024-context SigLIP2 + ELF-L pipeline.
# Runs stage 1 (vision projector warmup) and then stage 2 (LLaVA instruct)
# inside the same HOPE app so stage 2 consumes the freshly written stage-1 ckpt.
set -euo pipefail

STAGE1_CONFIG="${1:-src/configs/training_configs/train_llava_siglip2_warmup_ELF-L_hope_1024_vt196.yml}"
STAGE2_CONFIG="${2:-src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_hope_1024_vt196.yml}"
PROJECT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro"
CONDA_ROOT="/usr/local/conda"
ELF_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin"

source "${CONDA_ROOT}/bin/activate" "${ELF_ENV}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/submit/vendor:${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export ELF_IMPORT_STAGGER_SEC="0.25"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DIST_INIT_TIMEOUT_MIN=10
export NCCL_DEBUG=WARN

read_config_value() {
  python3 - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
print(cfg[sys.argv[2]])
PY
}

STAGE1_OUTPUT_DIR="$(read_config_value "${STAGE1_CONFIG}" output_dir)"
STAGE2_OUTPUT_DIR="$(read_config_value "${STAGE2_CONFIG}" output_dir)"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-${PROJECT_DIR}/outputs/elf_l-llava-siglip2-1024-vt196-pipeline-hope-32gpu/launch_logs}"
mkdir -p "${PIPELINE_LOG_DIR}"
LAUNCH_LOG="${PIPELINE_LOG_DIR}/worker_${AFO_TASK_ID:-unknown}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "===== ELF-L LLaVA 1024 pipeline launch $(date '+%F %T') ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "stage1_config=${STAGE1_CONFIG}"
echo "stage1_output_dir=${STAGE1_OUTPUT_DIR}"
echo "stage2_config=${STAGE2_CONFIG}"
echo "stage2_output_dir=${STAGE2_OUTPUT_DIR}"
echo "conda_env=${CONDA_PREFIX:-}"
echo "python=$(command -v python3)"
echo "AFO_TASK_ID=${AFO_TASK_ID:-}"
echo "AFO_ENV_CLUSTER_SPEC=${AFO_ENV_CLUSTER_SPEC:-}"
python3 -u - <<'PY'
import sys
print("probe_python_start", flush=True)
print("python_version", sys.version, flush=True)
print("python_executable", sys.executable, flush=True)
for mod in ("torch", "transformers", "datasets", "muon", "PIL"):
    try:
        m = __import__(mod)
        print(f"import {mod}", getattr(m, "__version__", "ok"), flush=True)
    except Exception as e:
        print(f"import {mod} FAILED: {e}", flush=True)
PY

if [ ! -f "${PROJECT_DIR}/submit/vendor/muon.py" ]; then
  echo "vendor muon.py missing; installing muon-optimizer"
  python3 -m pip install --no-cache-dir "muon-optimizer==0.1.0"
fi

TORCHRUN_FLAGS="$(python3 submit/hope_run_torch_distribute.py)"
echo "torchrun_flags=${TORCHRUN_FLAGS}"

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  echo "===== ${stage_name} torchrun start $(date '+%F %T') ====="
  HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run ${TORCHRUN_FLAGS} src/train.py --config "${config_path}"
  echo "===== ${stage_name} torchrun done $(date '+%F %T') ====="
}

wait_for_checkpoint() {
  local ckpt_dir="$1"
  local max_wait_sec="${2:-1800}"
  local waited=0
  while true; do
    if find "${ckpt_dir}" -maxdepth 1 -name 'checkpoint_*' -type f | grep -q .; then
      find "${ckpt_dir}" -maxdepth 1 -name 'checkpoint_*' -type f -printf '%T@ %p\n' | sort -n | tail -n 1
      return 0
    fi
    if [ "${waited}" -ge "${max_wait_sec}" ]; then
      echo "Timed out waiting for checkpoint in ${ckpt_dir}" >&2
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

run_stage "stage1_vision_warmup_1024_vt196" "${STAGE1_CONFIG}"
echo "===== waiting for stage1 checkpoint $(date '+%F %T') ====="
wait_for_checkpoint "${STAGE1_OUTPUT_DIR}" 1800
sleep 10
run_stage "stage2_mm_instruct_1024_vt196" "${STAGE2_CONFIG}"
echo "===== ELF-L LLaVA 1024 pipeline finished $(date '+%F %T') ====="
