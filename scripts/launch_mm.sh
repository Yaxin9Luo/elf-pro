#!/usr/bin/env bash
# Local launcher for the SigLIP2 + ELF multimodal pipeline.
#
# Stage 1 (vision projector warmup, 2x H100):
#     bash scripts/launch_mm.sh stage1
#
# Stage 2 (full instruct finetune, needs all 5 LLaVA-1.5 image subsets on disk):
#     bash scripts/launch_mm.sh stage2
#
# Smoke variants (single-GPU, end-to-end path, no quality target):
#     bash scripts/launch_mm.sh stage1 --smoke    # batch=8 + log_freq=10 over the full 558k JSON
#     bash scripts/launch_mm.sh stage2 --smoke    # uses the 200-record smoke JSON (ocr_vqa+textvqa only)
#
# H200 cluster overrides:
#     NGPU=8 bash scripts/launch_mm.sh stage1 --config_override global_batch_size=512
#
# Pass-through extra args land on train.py, e.g. --config_override use_compile=true.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash scripts/launch_mm.sh <stage1|stage2> [--smoke] [extra args...]"
    exit 1
fi

STAGE=$1
shift

EXTRA=()
SMOKE=0
for arg in "$@"; do
    if [[ "$arg" == "--smoke" ]]; then
        SMOKE=1
    else
        EXTRA+=("$arg")
    fi
done

case "$STAGE" in
    stage1)
        if [[ "$SMOKE" == "1" ]]; then
            CONFIG=src/configs/training_configs/train_llava_siglip2_warmup_ELF-L_local.yml
        else
            CONFIG=src/configs/training_configs/train_llava_siglip2_warmup_ELF-L_local.yml
        fi ;;
    stage2)
        if [[ "$SMOKE" == "1" ]]; then
            CONFIG=src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_smoke.yml
        else
            CONFIG=src/configs/training_configs/train_llava_siglip2_instruct_ELF-L_local.yml
        fi ;;
    *) echo "Unknown stage: $STAGE (expected 'stage1' or 'stage2')"; exit 1 ;;
esac

# Hugging Face / tokenizer plumbing — keep runtime fully offline. Local model dirs
# below mean transformers should never reach out, but we set the env vars anyway
# so transient HEAD requests don't stall the run.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# T5 / GPT2 weights are already in the HF cache above; don't try to fetch.
unset http_proxy https_proxy

NGPU=${NGPU:-$(command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "$SMOKE" == "1" ]]; then
    NGPU=1
    if [[ "$STAGE" == "stage1" ]]; then
        # stage1 smoke: full 558k JSON but tiny-batch single-GPU run.
        EXTRA+=(--config_override "global_batch_size=8")
        EXTRA+=(--config_override "log_freq=10")
        EXTRA+=(--config_override "save_freq=1")
        EXTRA+=(--config_override "num_workers=2")
        EXTRA+=(--config_override "gradient_checkpointing=true")
        echo "[launch_mm] stage1 smoke: NGPU=1 global_batch_size=8 log_freq=10 (full 558k dataset)"
    else
        # stage2 smoke: smoke YAML already pins batch=8 + 200-record JSON; just force 1 GPU.
        echo "[launch_mm] stage2 smoke: NGPU=1, using smoke YAML ($CONFIG)"
    fi
fi

if [[ "$NGPU" == "1" && "$NNODES" == "1" ]]; then
    echo "[launch_mm] $STAGE single-process: python src/train.py --config $CONFIG ${EXTRA[*]:-}"
    exec python src/train.py --config "$CONFIG" "${EXTRA[@]+"${EXTRA[@]}"}"
fi

echo "[launch_mm] $STAGE torchrun nproc_per_node=$NGPU nnodes=$NNODES node_rank=$NODE_RANK"
exec torchrun \
    --nproc_per_node="$NGPU" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/train.py --config "$CONFIG" "${EXTRA[@]+"${EXTRA[@]}"}"
