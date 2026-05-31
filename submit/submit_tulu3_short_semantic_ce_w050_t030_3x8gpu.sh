#!/usr/bin/env bash
# Submit pmean=-3 semantic CE jobs for the three decoder_prob candidates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v hope >/dev/null 2>&1; then
  echo "hope CLI not found in PATH. Run this from a node with HOPE access." >&2
  exit 1
fi

jobs=(
  submit_elf_l_tulu3_short_semantic_ce_pmean_m3_dp025_w050_t030_8gpu.hope
  submit_elf_l_tulu3_short_semantic_ce_pmean_m3_dp035_w050_t030_8gpu.hope
  submit_elf_l_tulu3_short_semantic_ce_pmean_m3_dp050_w050_t030_8gpu.hope
)

for job in "${jobs[@]}"; do
  echo "===== hope run ${job} ====="
  hope run "${job}"
done
