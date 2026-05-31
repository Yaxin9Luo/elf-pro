#!/usr/bin/env bash
# Submit the two missing 8-GPU points for the pmean=-3 decoder_prob sweep.
# The 0.25 point is the existing pmean=-3 run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v hope >/dev/null 2>&1; then
  echo "hope CLI not found in PATH. Run this from a node with HOPE access." >&2
  exit 1
fi

jobs=(
  submit_elf_l_tulu3_short_ablate_pmean_m3_dp035_8gpu.hope
  submit_elf_l_tulu3_short_ablate_pmean_m3_dp050_8gpu.hope
)

for job in "${jobs[@]}"; do
  echo "===== hope run ${job} ====="
  hope run "${job}"
done
