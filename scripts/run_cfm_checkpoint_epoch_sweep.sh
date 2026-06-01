#!/usr/bin/env bash
# Run full CFM standard benchmarks over one checkpoint per training epoch.
#
# Defaults target the previous 20-epoch English Tulu3 Semantic CE SFT run.
# Override EPOCHS with a comma-separated list such as:
#   EPOCHS=1,2,3,5,8,10,12,15,18,20 bash scripts/run_cfm_checkpoint_epoch_sweep.sh
set -euo pipefail

CONFIG="${CONFIG:-src/configs/training_configs/train_tulu3_sft_english_semantic_ce_pmean_m3_dp050_w050_t030_ELF-L_hope_32gpu_20ep.yml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/elf_l-tulu3-english-semce-pmean_m3-dp050-w050-t030-20ep-hope-32gpu}"
RESULT_ROOT="${RESULT_ROOT:-eval_probes/standard_benchmarks/results/sft_tulu3_english_semce_20ep_epoch_sweep_full_32step}"
BENCH_DIR="${BENCH_DIR:-eval_probes/standard_benchmarks/data}"
SAMPLING_CONFIG="${SAMPLING_CONFIG:-eval_probes/sft_probe_sampling_32.yml}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_cfm_standard_benchmarks_sharded.sh}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
DEVICE="${DEVICE:-cuda}"
IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR:-third_party/google_ifeval_current}"
EPOCHS="${EPOCHS:-}"
FORCE="${FORCE:-0}"
USE_LOCK="${USE_LOCK:-1}"
EVAL_LOCK="${EVAL_LOCK:-/tmp/elf_pro_cfm_eval_gpu.lock}"

mkdir -p "${RESULT_ROOT}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

mapfile -t checkpoints < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "checkpoint_*" | sort -V)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoint_* files found under ${CHECKPOINT_DIR}" >&2
  exit 2
fi

epoch_requested() {
  local epoch="$1"
  if [[ -z "${EPOCHS}" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a epoch_list <<< "${EPOCHS}"
  for item in "${epoch_list[@]}"; do
    if [[ "${item}" == "${epoch}" ]]; then
      return 0
    fi
  done
  return 1
}

write_sweep_summary() {
  python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
metrics = [
    "ifeval_strict_prompt",
    "ifeval_strict_instruction",
    "boolq",
    "svamp",
    "truthfulqa_mc",
    "winogrande",
    "arc_challenge",
    "openbookqa",
    "hellaswag",
    "mmlu_pro",
    "gsm8k",
]

rows = []
for result_dir in sorted(result_root.glob("epoch_*")):
    aggregate = result_dir / "aggregate_summary.json"
    meta_path = result_dir / "sweep_meta.json"
    if not aggregate.is_file():
        continue
    scores = json.loads(aggregate.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    row = {
        "epoch": meta.get("epoch", result_dir.name),
        "checkpoint": meta.get("checkpoint", ""),
        "result_dir": str(result_dir),
    }
    for metric in metrics:
        score = (scores.get(metric) or {}).get("score")
        row[metric] = score
    rows.append(row)

summary_json = result_root / "epoch_sweep_summary.json"
summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# CFM Epoch Sweep Summary",
    "",
    "| epoch | checkpoint | " + " | ".join(metrics) + " |",
    "|---:|---|" + "|".join(["---:"] * len(metrics)) + "|",
]
for row in rows:
    values = []
    for metric in metrics:
        score = row.get(metric)
        values.append("-" if score is None else f"{float(score) * 100:.2f}")
    lines.append(
        f"| {row['epoch']} | {Path(row['checkpoint']).name} | "
        + " | ".join(values)
        + " |"
    )

(result_root / "epoch_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {summary_json} with {len(rows)} evaluated epochs")
PY
}

run_one_checkpoint() {
  local epoch="$1"
  local checkpoint="$2"
  local ckpt_name
  ckpt_name="$(basename "${checkpoint}")"
  local out_dir="${RESULT_ROOT}/epoch_$(printf "%02d" "${epoch}")_${ckpt_name}"
  local marker="${out_dir}/.benchmark_done"

  if [[ "${FORCE}" != "1" && -f "${marker}" && -f "${out_dir}/aggregate_summary.json" ]]; then
    echo "Skipping epoch ${epoch} (${ckpt_name}); benchmark already done"
    return 0
  fi

  mkdir -p "${out_dir}"
  python3 - "${out_dir}/sweep_meta.json" "${epoch}" "${checkpoint}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = {
    "epoch": int(sys.argv[2]),
    "checkpoint": sys.argv[3],
    "config": sys.argv[4],
}
path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  echo "===== epoch ${epoch}: ${checkpoint} -> ${out_dir} ====="
  local lock_fd=""
  if [[ "${USE_LOCK}" == "1" ]]; then
    echo "Waiting for eval GPU lock: ${EVAL_LOCK}"
    exec {lock_fd}>"${EVAL_LOCK}"
    flock "${lock_fd}"
    echo "Acquired eval GPU lock"
  fi

  CONFIG="${CONFIG}" \
  CHECKPOINT="${checkpoint}" \
  BENCH_DIR="${BENCH_DIR}" \
  OUT_DIR="${out_dir}" \
  SAMPLING_CONFIG="${SAMPLING_CONFIG}" \
  MAX_EXAMPLES="${MAX_EXAMPLES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
  DEVICE="${DEVICE}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR}" \
    bash "${RUNNER_SCRIPT}"

  if [[ "${USE_LOCK}" == "1" ]]; then
    flock -u "${lock_fd}"
    eval "exec ${lock_fd}>&-"
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${marker}"
  write_sweep_summary
}

main() {
  local epoch=0
  local checkpoint
  for checkpoint in "${checkpoints[@]}"; do
    epoch=$((epoch + 1))
    if ! epoch_requested "${epoch}"; then
      continue
    fi
    run_one_checkpoint "${epoch}" "${checkpoint}"
  done
  write_sweep_summary
}

main
