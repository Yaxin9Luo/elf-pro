#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2-large}"
BENCH_DIR="${BENCH_DIR:-eval_probes/standard_benchmarks/data}"
OUT_DIR="${OUT_DIR:-eval_probes/standard_benchmarks/results/ar_gpt2_large_full_32step}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
USE_SLOW_TOKENIZER="${USE_SLOW_TOKENIZER:-0}"

mkdir -p "${OUT_DIR}"

local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_args+=(--local_files_only)
fi
if [[ "${USE_SLOW_TOKENIZER}" == "1" ]]; then
  local_args+=(--use_slow_tokenizer)
fi

for bench_file in "${BENCH_DIR}"/*.jsonl; do
  name="$(basename "${bench_file}" .jsonl)"
  echo "===== ${name} ====="
  python3 eval_probes/run_ar_standard_benchmark.py \
    --model "${MODEL}" \
    --benchmark_file "${bench_file}" \
    --output_jsonl "${OUT_DIR}/${name}.jsonl" \
    --summary_json "${OUT_DIR}/${name}.summary.json" \
    --prompt_format "${PROMPT_FORMAT}" \
    --max_examples "${MAX_EXAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    "${local_args[@]}"

  if [[ "${name}" == "ifeval" && -n "${IFEVAL_GOOGLE_DIR}" ]]; then
    if [[ "${MAX_EXAMPLES}" != "0" ]]; then
      python3 eval_probes/score_ifeval_official.py \
        --benchmark_jsonl "${bench_file}" \
        --generations_jsonl "${OUT_DIR}/${name}.jsonl" \
        --google_research_dir "${IFEVAL_GOOGLE_DIR}" \
        --output_dir "${OUT_DIR}/ifeval_official" \
        --allow_partial
    else
      python3 eval_probes/score_ifeval_official.py \
        --benchmark_jsonl "${bench_file}" \
        --generations_jsonl "${OUT_DIR}/${name}.jsonl" \
        --google_research_dir "${IFEVAL_GOOGLE_DIR}" \
        --output_dir "${OUT_DIR}/ifeval_official"
    fi
  fi
done

python3 eval_probes/summarize_standard_benchmark_results.py \
  --result_dir "${OUT_DIR}"
