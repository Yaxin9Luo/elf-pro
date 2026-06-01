#!/usr/bin/env bash
# Submit the 32GPU English Tulu3 SFT run with time-scheduled multi-time Semantic CE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v hope >/dev/null 2>&1; then
  echo "hope CLI not found in PATH. Run this from a node with HOPE access." >&2
  exit 1
fi

hope run submit_elf_l_tulu3_sft_english_semce_schedule_multit_pmean_m3_dp050_w050_t030_20ep_32gpu.hope
