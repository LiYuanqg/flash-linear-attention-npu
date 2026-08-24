#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${ROOT}"
rm -rf outputs
PY=${PY:-$HOME/.venvs/fla/bin/python3}
export PYTHONPATH=${HOME}/golden/fla:${PYTHONPATH:-}
export MPLBACKEND=Agg
export CT=${CT:-$HOME/.venvs/fla/bin/ct}

"${PY}" run_cpu_golden.py --compute-precision fp32 --safe-gate --out-dir outputs/cpu_fp32 --save-inputs
"${PY}" run_triton.py --backend staged --safe-gate --out-dir outputs/triton_staged
"${PY}" run_triton.py --backend fla --safe-gate --out-dir outputs/triton_fla
"${PY}" compare_stats.py --real-dir outputs/cpu_fp32 --expect-dir outputs/triton_staged | tee outputs/compare_staged.txt
"${PY}" compare_stats.py --real-dir outputs/cpu_fp32 --expect-dir outputs/triton_fla | tee outputs/compare_fla.txt
"${PY}" compare_stats.py --real-dir outputs/triton_staged --expect-dir outputs/triton_fla | tee outputs/compare_staged_vs_fla.txt
bash run_ct_viz.sh outputs/cpu_fp32 outputs/triton_staged outputs/ct_staged
bash run_ct_viz.sh outputs/cpu_fp32 outputs/triton_fla outputs/ct_fla gk qg kg w u
echo done
