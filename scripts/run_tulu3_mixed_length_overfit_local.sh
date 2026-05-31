#!/usr/bin/env bash
# Run the clean-English Tulu3 mixed-length overfit experiment on the dev server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATA_DIR="${DATA_DIR:-/tmp/elf-pro-cache/data/tulu3_mixed_length_clean_en_10k_t5_1024_eos}"
export PROBE_JSONL="${PROBE_JSONL:-/tmp/elf-pro-cache/data/tulu3_mixed_length_clean_en_10k_probe.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/elf-pro-cache/tulu3_mixed_length_clean_en_10k_ELF-L_t5_1024_from_sft5402}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
export EPOCHS="${EPOCHS:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export MASTER_PORT="${MASTER_PORT:-29646}"

bash "${REPO_ROOT}/scripts/run_tulu3_short_answer_overfit_local.sh" "$@"
