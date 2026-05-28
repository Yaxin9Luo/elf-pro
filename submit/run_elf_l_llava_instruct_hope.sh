#!/usr/bin/env bash
# HOPE worker.script for the ELF-L SigLIP2 stage-2 instruct run on sh02.
# Mirrors run_elf_l_hope.sh's plumbing (shared conda env, vendored muon,
# offline HF, hope_run_torch_distribute) but points at the multimodal YAML.
set -euo pipefail

CONFIG_PATH="${1:-src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_hope_32gpu.yml}"
PROJECT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03/projects/elf-pro"
CONDA_ROOT="/usr/local/conda"
ELF_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/envs/audio_jmh2_clone_yaxin"

source "${CONDA_ROOT}/bin/activate" "${ELF_ENV}"
cd "${PROJECT_DIR}"

# submit/vendor first so the vendored muon-optimizer copy wins; src/ for bare
# `from modules.model import ...` imports.
export PYTHONPATH="${PROJECT_DIR}/submit/vendor:${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export ELF_IMPORT_STAGGER_SEC="0.25"
export TOKENIZERS_PARALLELISM=false
# Cluster has no internet — make every HF call hit the local mirror dirs we
# baked into the YAML (encoder_model_name, vision_encoder_model_name, etc.).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# NCCL / distributed safety knobs cribbed from run_elf_l_hope.sh.
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

echo "===== ELF-L LLaVA stage2 launch $(date '+%F %T') ====="
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

echo "===== torchrun start $(date '+%F %T') ====="
TORCHRUN_FLAGS="$(python3 submit/hope_run_torch_distribute.py)"
echo "torchrun_flags=${TORCHRUN_FLAGS}"
HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run ${TORCHRUN_FLAGS} src/train.py --config "${CONFIG_PATH}"
