#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-src/configs/training_configs/train_tulu3_sft_english_semantic_ce_pmean_m3_dp050_w050_t030_ELF-L_hope_32gpu_20ep.yml}"
CHECKPOINT="${CHECKPOINT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/elf-pro/outputs/elf_l-owt-hf-torch/checkpoint_57051}"
BENCH_DIR="${BENCH_DIR:-eval_probes/standard_benchmarks/data}"
OUT_DIR="${OUT_DIR:-eval_probes/standard_benchmarks/results/pretrained_owt_ckpt57051_full_sharded}"
SAMPLING_CONFIG="${SAMPLING_CONFIG:-eval_probes/sft_probe_sampling_32.yml}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
IFEVAL_GOOGLE_DIR="${IFEVAL_GOOGLE_DIR:-}"

IFS=',' read -r -a device_list <<< "${CUDA_DEVICES}"
num_shards="${#device_list[@]}"
if [[ "${num_shards}" -lt 1 ]]; then
  echo "CUDA_DEVICES must contain at least one device id" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/shards"

for bench_file in "${BENCH_DIR}"/*.jsonl; do
  name="$(basename "${bench_file}" .jsonl)"
  echo "===== ${name} (${num_shards} shards) ====="
  shard_jsonls=()
  pids=()
  for shard_index in "${!device_list[@]}"; do
    gpu="${device_list[$shard_index]}"
    shard_jsonl="${OUT_DIR}/shards/${name}.shard_${shard_index}.jsonl"
    shard_summary="${OUT_DIR}/shards/${name}.shard_${shard_index}.summary.json"
    shard_jsonls+=("${shard_jsonl}")
    (
      CUDA_VISIBLE_DEVICES="${gpu}" python3 eval_probes/run_cfm_standard_benchmark.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --benchmark_file "${bench_file}" \
        --sampling_config "${SAMPLING_CONFIG}" \
        --output_jsonl "${shard_jsonl}" \
        --summary_json "${shard_summary}" \
        --max_examples "${MAX_EXAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --max_input_length "${MAX_INPUT_LENGTH}" \
        --device "${DEVICE}" \
        --num_shards "${num_shards}" \
        --shard_index "${shard_index}"
    ) &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one shard failed for ${name}" >&2
    exit 1
  fi

  python3 eval_probes/merge_standard_benchmark_shards.py \
    --shard_jsonl "${shard_jsonls[@]}" \
    --output_jsonl "${OUT_DIR}/${name}.jsonl" \
    --summary_json "${OUT_DIR}/${name}.summary.json"

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
