#!/usr/bin/env bash
set -euo pipefail

SFT_OUT_DIR="${SFT_OUT_DIR:-outputs/langflow-owt-tulu3-english-sft-20ep-32gpu}"
export CHECKPOINT="${CHECKPOINT:-${SFT_OUT_DIR}}"
export OUT_DIR="${OUT_DIR:-eval_probes/standard_benchmarks/results/langflow_owt_tulu3_english_sft_latest}"
export PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
export ADD_BOS="${ADD_BOS:-1}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-768}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

bash scripts/run_langflow_standard_benchmarks_sharded.sh
