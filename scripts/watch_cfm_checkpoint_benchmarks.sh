#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONFIG:-}" ]]; then
  echo "CONFIG is required" >&2
  exit 2
fi

if [[ -z "${CKPT_GLOB:-}" ]]; then
  echo "CKPT_GLOB is required, for example: /path/to/output/checkpoint_*" >&2
  exit 2
fi

BENCH_DIR="${BENCH_DIR:-eval_probes/standard_benchmarks/data}"
RESULT_ROOT="${RESULT_ROOT:-eval_probes/standard_benchmarks/results}"
SAMPLING_CONFIG="${SAMPLING_CONFIG:-eval_probes/sft_probe_sampling_32.yml}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
DEVICE="${DEVICE:-cuda}"
IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR:-}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_cfm_standard_benchmarks.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RUNS="${MAX_RUNS:-0}"

latest_checkpoint() {
  find "$(dirname "${CKPT_GLOB}")" -maxdepth 1 -type f -name "$(basename "${CKPT_GLOB}")" \
    | sort -V \
    | tail -1
}

runs=0
while true; do
  checkpoint="$(latest_checkpoint || true)"
  if [[ -z "${checkpoint}" ]]; then
    echo "No checkpoint found for ${CKPT_GLOB}; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  name="$(basename "${checkpoint}")"
  out_dir="${RESULT_ROOT}/${name}"
  marker="${out_dir}/.benchmark_done"
  if [[ -f "${marker}" ]]; then
    echo "Already evaluated ${checkpoint}; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  echo "Evaluating ${checkpoint} -> ${out_dir}"
  CONFIG="${CONFIG}" \
  CHECKPOINT="${checkpoint}" \
  BENCH_DIR="${BENCH_DIR}" \
  OUT_DIR="${out_dir}" \
  SAMPLING_CONFIG="${SAMPLING_CONFIG}" \
  MAX_EXAMPLES="${MAX_EXAMPLES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
  DEVICE="${DEVICE}" \
  IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR}" \
    bash "${RUNNER_SCRIPT}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${marker}"

  runs=$((runs + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${runs}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}"
    exit 0
  fi
done
