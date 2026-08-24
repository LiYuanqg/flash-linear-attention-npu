#!/usr/bin/env bash
set -euo pipefail

# CPU golden is real; Triton is expect.
# Usage: bash run_ct_viz.sh <cpu_dir> <triton_dir> <out_dir> [names...]

CPU_DIR=${1:?cpu dir}
TRI_DIR=${2:?triton dir}
OUT_DIR=${3:?out dir}
shift 3
if [ "$#" -gt 0 ]; then
    NAMES=("$@")
else
    NAMES=(g_corr gk qg kbg vb kg w u)
fi

mkdir -p "${OUT_DIR}"
CT=${CT:-$HOME/.local/bin/ct}

for name in "${NAMES[@]}"; do
    cpu="${CPU_DIR}/${name}.pt"
    tri="${TRI_DIR}/${name}.pt"
    if [ ! -f "${cpu}" ] || [ ! -f "${tri}" ]; then
        echo "skip ${name}: missing ${cpu} or ${tri}"
        continue
    fi
    echo "ct viz ${name}"
    "${CT}" viz "${cpu}" "${tri}" \
        --out_dir "${OUT_DIR}" \
        --name "${name}" \
        -sc 100000 \
        -wl 1
done
