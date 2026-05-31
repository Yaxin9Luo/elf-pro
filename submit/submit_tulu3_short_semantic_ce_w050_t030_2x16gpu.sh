#!/usr/bin/env bash
# Submit the two 16GPU semantic CE jobs:
# dp=0.25 control and dp=0.50 main.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v hope >/dev/null 2>&1; then
  echo "hope CLI not found in PATH. Run this from a node with HOPE access." >&2
  exit 1
fi

jobs=(
  submit_elf_l_tulu3_short_semantic_ce_pmean_m3_dp025_w050_t030_16gpu.hope
  submit_elf_l_tulu3_short_semantic_ce_pmean_m3_dp050_w050_t030_16gpu.hope
)

for job in "${jobs[@]}"; do
  echo "===== hope run ${job} ====="
  hope run "${job}"
done
