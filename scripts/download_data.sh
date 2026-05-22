#!/bin/bash
# Resilient OWT downloader with retry + resume.
# Usage: bash scripts/download_data.sh [target_dir]
set -u

TARGET="${1:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/data/elf-data/openwebtext-t5}"
REPO="embedded-language-flows/openwebtext-t5"

export http_proxy=http://10.70.16.106:3128
export https_proxy=http://10.70.16.106:3128
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,sankuai.com,aliyun.com,mirrors.aliyun.com"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$TARGET"
echo "Target: $TARGET"

attempt=0
max_attempts=20
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    echo "=== attempt $attempt / $max_attempts ==="
    /usr/local/conda/bin/python3 - <<PYEOF
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        repo_id="$REPO",
        repo_type="dataset",
        local_dir="$TARGET",
        max_workers=4,
        etag_timeout=60,
    )
    print("DONE")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "Download complete."
        # Verify file count == 75 arrows + dataset_info.json + state.json + .gitattributes
        n=$(ls "$TARGET" | grep -c "\.arrow$")
        echo "Found $n .arrow files (expected 75)"
        if [ "$n" = "75" ]; then
            exit 0
        fi
        echo "File count mismatch, retrying..."
    fi
    sleep 15
done

echo "FAILED after $max_attempts attempts"
exit 1
