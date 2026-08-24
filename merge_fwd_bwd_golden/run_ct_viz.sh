#!/usr/bin/env bash
set -euo pipefail

CPU_DIR=${1:?cpu dir}
TRI_DIR=${2:?triton dir}
OUT_DIR=${3:?out dir}
shift 3
if [ "$#" -gt 0 ]; then
    NAMES=("$@")
else
    NAMES=()
    for path in "${CPU_DIR}"/*.pt; do
        name=$(basename "${path}" .pt)
        if [ -f "${TRI_DIR}/${name}.pt" ]; then
            NAMES+=("${name}")
        fi
    done
fi

mkdir -p "${OUT_DIR}"
CT=${CT:-$HOME/.local/bin/ct}

for name in "${NAMES[@]}"; do
    echo "ct viz ${name}"
    "${CT}" viz "${CPU_DIR}/${name}.pt" "${TRI_DIR}/${name}.pt" \
        --out_dir "${OUT_DIR}" \
        --name "${name}" \
        -sc 100000 \
        -wl 1
done
