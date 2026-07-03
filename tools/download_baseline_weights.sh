#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/code/baselines}"
cd "$ROOT"

source /root/conda/etc/profile.d/conda.sh
conda activate m2a_env

export https_proxy="${https_proxy:-http://21.249.84.64:3128}"
export http_proxy="${http_proxy:-http://21.249.84.64:3128}"
export all_proxy="${all_proxy:-http://21.249.84.64:3128}"

mkdir -p _weights_try/pebal _weights_try/rpl_corocl _weights_try/uno

download_drive_file() {
    local label="$1"
    local url="$2"
    local output="$3"
    local min_bytes="$4"

    echo "== ${label} =="
    if [[ -f "$output" ]]; then
        local size
        size="$(stat -c '%s' "$output")"
        if (( size >= min_bytes )); then
            echo "skip existing $output ($size bytes)"
            return 0
        fi
        mv "$output" "${output}.incomplete.$(date +%s)"
    fi

    if ! gdown --fuzzy --no-cookies "$url" -O "$output"; then
        echo "FAILED ${label}"
        return 0
    fi

    stat -c 'downloaded %n %s bytes' "$output"
}

echo "START $(date)"

download_drive_file \
    "PEBAL best_ad_ckpt" \
    "https://drive.google.com/file/d/12CebI1TlgF724-xvI3vihjbIPPn5Icpm/view?usp=sharing" \
    "_weights_try/pebal/best_ad_ckpt.pth" \
    100000000

download_drive_file \
    "RPL+CoroCL rev3" \
    "https://drive.google.com/file/d/1fSn5xJZWqkbFZlhH1qPpbLvKh8dQS7yl/view?usp=sharing" \
    "_weights_try/rpl_corocl/rev3.pth" \
    900000000

download_drive_file \
    "UNO ADE20K negatives" \
    "https://drive.google.com/file/d/1ablD-t34MXcP-oSSzSq0-TNz0AxKtp_m/view?usp=sharing" \
    "_weights_try/uno/uno_ade.pth" \
    2500000000

download_drive_file \
    "UNO synthetic negatives" \
    "https://drive.google.com/file/d/108CHRZFWTnDBonQv2yRjRL3JNj4_y47E/view?usp=sharing" \
    "_weights_try/uno/uno_synthetic.pth" \
    2500000000

download_drive_file \
    "UNO DenseFlow" \
    "https://drive.google.com/file/d/1vS7K2irT2Gxh_8UQ9Aw1X5t5l6tG0Eol/view?usp=sharing" \
    "_weights_try/uno/denseflow.pth" \
    100000000

echo "DONE $(date)"
find _weights_try -maxdepth 3 -type f -printf '%p %s\n' | sort
