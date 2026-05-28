#!/usr/bin/env bash
# Mirror everything the HOPE stage-2 job needs from the dev box (hldy mount)
# to the sh02 mount. Run this *before* submitting submit_elf_l_llava_instruct_32gpu.hope,
# and re-run after every code edit / new stage1 checkpoint.
#
# Usage:
#     bash submit/sync_to_sh02.sh                # full sync
#     bash submit/sync_to_sh02.sh --code-only    # skip data + ckpts (fast iteration)
#     bash submit/sync_to_sh02.sh --dry-run      # show what would change
set -euo pipefail

DEV_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo"
SH02_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/luoyaxin03"

DEV_PROJECT="${DEV_ROOT}/projects/elf-pro"
SH02_PROJECT="${SH02_ROOT}/projects/elf-pro"

DEV_DATA="${DEV_ROOT}/data"
SH02_DATA="${SH02_PROJECT}/data"

CODE_ONLY=0
DRY=""
for arg in "$@"; do
    case "$arg" in
        --code-only) CODE_ONLY=1 ;;
        --dry-run) DRY="--dry-run" ;;
        *) echo "unknown flag: $arg"; exit 1 ;;
    esac
done

mkdir -p "${SH02_PROJECT}" "${SH02_DATA}/playground/data" "${SH02_PROJECT}/models" "${SH02_PROJECT}/outputs"

echo "===== sync code: ${DEV_PROJECT}/ -> ${SH02_PROJECT}/"
rsync -av ${DRY} \
    --exclude='.git' \
    --exclude='outputs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='wandb' \
    "${DEV_PROJECT}/" "${SH02_PROJECT}/"

if [[ "$CODE_ONLY" == "1" ]]; then
    echo "code-only sync done; data + ckpts skipped."
    exit 0
fi

echo "===== sync stage1 ckpt dir (skip if missing) ====="
STAGE1_LOCAL="${DEV_PROJECT}/outputs/elf_l-llava-siglip2-warmup-local"
STAGE1_REMOTE="${SH02_PROJECT}/outputs/elf_l-llava-siglip2-warmup"
if [[ -d "${STAGE1_LOCAL}" ]]; then
    rsync -av ${DRY} "${STAGE1_LOCAL}/" "${STAGE1_REMOTE}/"
else
    echo "  ${STAGE1_LOCAL} missing — stage1 hasn't produced a checkpoint yet."
fi

echo "===== sync mix665k JSON + image subsets ====="
rsync -av ${DRY} \
    "${DEV_DATA}/playground/data/llava_v1_5_mix665k.json" \
    "${SH02_DATA}/playground/data/"

for sub in coco gqa ocr_vqa textvqa vg; do
    src="${DEV_DATA}/playground/data/${sub}"
    if [[ -d "$src" ]]; then
        echo "  syncing ${sub}/ ..."
        rsync -a ${DRY} "${src}/" "${SH02_DATA}/playground/data/${sub}/"
    else
        echo "  ${sub}/ missing — skipping (still downloading?)"
    fi
done

echo "===== sync model weights (t5-small + siglip2-base) ====="
# t5-small comes from the HF cache rather than the project dir.
T5_CACHE_LATEST=$(ls -d /home/hadoop-aipnlp/.cache/huggingface/hub/models--t5-small/snapshots/*/ 2>/dev/null | head -1 || true)
if [[ -n "${T5_CACHE_LATEST}" ]]; then
    rsync -avL ${DRY} "${T5_CACHE_LATEST}" "${SH02_PROJECT}/models/t5-small/"
else
    echo "  WARN: t5-small not found in HF cache — populate ${SH02_PROJECT}/models/t5-small/ manually."
fi

if [[ -d "${DEV_PROJECT}/models/siglip2-base-patch16-224" ]]; then
    rsync -av ${DRY} \
        "${DEV_PROJECT}/models/siglip2-base-patch16-224/" \
        "${SH02_PROJECT}/models/siglip2-base-patch16-224/"
fi

echo
echo "===== sync done. Submit from sh02 with:"
echo "  cd ${SH02_PROJECT}/submit"
echo "  hope submit submit_elf_l_llava_instruct_32gpu.hope"
