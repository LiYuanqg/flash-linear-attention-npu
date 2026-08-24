#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${ROOT}"
rm -rf outputs
PY=${PY:-$HOME/.venvs/fla/bin/python3}
export PYTHONPATH=${HOME}/golden/fla:${PYTHONPATH:-}
export MPLBACKEND=Agg
export CT=${CT:-$HOME/.venvs/fla/bin/ct}

run_case() {
    local name=$1
    shift
    echo "===== ${name} ====="
    "${PY}" run_cpu_golden.py --out-dir "outputs/${name}_cpu" "$@"
    "${PY}" run_triton.py --backend staged --out-dir "outputs/${name}_staged" "$@"
    if [[ " $* " != *" --has-h0 "* ]]; then
        "${PY}" run_triton.py --backend fla --out-dir "outputs/${name}_fla" "$@"
    fi
    "${PY}" compare_stats.py --real-dir "outputs/${name}_cpu" --expect-dir "outputs/${name}_staged" \
        | tee "outputs/${name}_compare_staged.txt"
    if [ -d "outputs/${name}_fla" ]; then
        "${PY}" compare_stats.py --real-dir "outputs/${name}_cpu" --expect-dir "outputs/${name}_fla" \
            | tee "outputs/${name}_compare_fla.txt"
    fi
    bash run_ct_viz.sh "outputs/${name}_cpu" "outputs/${name}_staged" "outputs/ct_${name}_staged"
}

run_case fwd "$@"
run_case bwd --direction bwd
run_case fwd_h0 --has-h0
echo "done"
