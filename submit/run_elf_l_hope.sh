#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-src/configs/training_configs/train_owt_ELF-L_hope_8gpu.yml}"
PROJECT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/luoyaxin03/projects/elf-pro"
CONDA_ROOT="/usr/local/conda"
ELF_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin"
MUON_LOCK="${PROJECT_DIR}/.muon_install.lock"

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

OUTPUT_DIR="$(python3 - "${CONFIG_PATH}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
print(cfg["output_dir"])
PY
)"
LAUNCH_LOG_DIR="${OUTPUT_DIR}/launch_logs"
mkdir -p "${LAUNCH_LOG_DIR}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/worker_${AFO_TASK_ID:-unknown}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "===== ELF HOPE launch $(date '+%F %T') ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "config=${CONFIG_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "conda_env=${CONDA_PREFIX:-}"
echo "python=$(command -v python3)"
echo "AFO_TASK_ID=${AFO_TASK_ID:-}"
echo "AFO_ENV_CLUSTER_SPEC=${AFO_ENV_CLUSTER_SPEC:-}"
python3 -u - <<'PY'
import sys
print("probe_python_start", flush=True)
print("python_version", sys.version, flush=True)
print("python_executable", sys.executable, flush=True)
PY

if [ ! -f "${PROJECT_DIR}/submit/vendor/muon.py" ]; then
  echo "vendor muon.py missing; installing muon-optimizer"
  python3 -m pip install --no-cache-dir "muon-optimizer==0.1.0"
fi

echo "===== torchrun start $(date '+%F %T') ====="
TORCHRUN_FLAGS="$(python3 submit/hope_run_torch_distribute.py)"
echo "torchrun_flags=${TORCHRUN_FLAGS}"
HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run ${TORCHRUN_FLAGS} src/train.py --config "${CONFIG_PATH}"
