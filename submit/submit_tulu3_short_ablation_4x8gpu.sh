#!/usr/bin/env bash
# Submit the four single-node 8-GPU Tulu3 Short QA recipe ablations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v hope >/dev/null 2>&1; then
  echo "hope CLI not found in PATH. Run this from a node with HOPE access." >&2
  exit 1
fi

jobs=(
  submit_elf_l_tulu3_short_ablate_dp010_8gpu.hope
  submit_elf_l_tulu3_short_ablate_dp050_8gpu.hope
  submit_elf_l_tulu3_short_ablate_pmean_m3_8gpu.hope
  submit_elf_l_tulu3_short_ablate_uniform_time_8gpu.hope
)

for job in "${jobs[@]}"; do
  echo "===== hope run ${job} ====="
  hope run "${job}"
done
