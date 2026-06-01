#!/usr/bin/env bash
# Wait for a HOPE run to finish, then benchmark the latest checkpoint.
set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi
if [[ -z "${CONFIG:-}" ]]; then
  echo "CONFIG is required" >&2
  exit 2
fi
if [[ -z "${CKPT_GLOB:-}" ]]; then
  echo "CKPT_GLOB is required" >&2
  exit 2
fi
if [[ -z "${RESULT_ROOT:-}" ]]; then
  echo "RESULT_ROOT is required" >&2
  exit 2
fi

SAMPLING_CONFIG="${SAMPLING_CONFIG:-eval_probes/sft_probe_sampling_32.yml}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR:-third_party/google_ifeval_current}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_cfm_standard_benchmarks_sharded.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RUNS="${MAX_RUNS:-1}"
USE_LOCK="${USE_LOCK:-1}"
EVAL_LOCK="${EVAL_LOCK:-/tmp/elf_pro_cfm_eval_gpu.lock}"

read_state() {
  local status_json
  status_json="$(hope status --runid="${RUN_ID}" --json --nocache || true)"
  STATUS_JSON="${status_json}" python3 - <<'PY'
import json
import os

try:
    data = json.loads(os.environ.get("STATUS_JSON", "{}"))
    apps = data.get("apps_info") or []
    print(apps[0].get("state", "UNKNOWN") if apps else "UNKNOWN")
except Exception:
    print("UNKNOWN")
PY
}

run_benchmark_once() {
  CONFIG="${CONFIG}" \
  CKPT_GLOB="${CKPT_GLOB}" \
  RESULT_ROOT="${RESULT_ROOT}" \
  SAMPLING_CONFIG="${SAMPLING_CONFIG}" \
  MAX_EXAMPLES="${MAX_EXAMPLES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
  DEVICE="${DEVICE}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR}" \
  RUNNER_SCRIPT="${RUNNER_SCRIPT}" \
  POLL_SECONDS="${POLL_SECONDS}" \
  MAX_RUNS="${MAX_RUNS}" \
    bash scripts/watch_cfm_checkpoint_benchmarks.sh
}

while true; do
  state="$(read_state)"
  echo "$(date +%Y-%m-%dT%H:%M:%S%z) run_id=${RUN_ID} state=${state}"
  case "${state}" in
    SUCCEEDED)
      if [[ "${USE_LOCK}" == "1" ]]; then
        echo "HOPE job succeeded; waiting for eval GPU lock: ${EVAL_LOCK}"
        exec 9>"${EVAL_LOCK}"
        flock 9
        echo "Acquired eval GPU lock; starting benchmark"
      else
        echo "HOPE job succeeded; starting benchmark"
      fi
      run_benchmark_once
      exit 0
      ;;
    FAILED|KILLED|STOPPED)
      echo "HOPE job ended with terminal state=${state}; not starting benchmark" >&2
      exit 1
      ;;
  esac
  sleep "${POLL_SECONDS}"
done
