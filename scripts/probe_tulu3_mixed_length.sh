#!/usr/bin/env bash
# Evaluate the clean-English Tulu3 mixed-length overfit checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PROBE_JSONL="${PROBE_JSONL:-/tmp/elf-pro-cache/data/tulu3_mixed_length_clean_en_10k_probe.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/elf-pro-cache/tulu3_mixed_length_clean_en_10k_ELF-L_t5_1024_from_sft5402}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-64}"

bash "${REPO_ROOT}/scripts/probe_tulu3_short_answer.sh" "$@"
