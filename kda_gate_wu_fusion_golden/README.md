# KDA `kda_gate_chunk_cumsum` + `recompute_w_u_fwd` stage golden

Local debug only. Do not copy machine names, accounts, or absolute host paths into public PRs.

## Stages

| Stage | Unit | Formula |
| --- | --- | --- |
| V0 | vector | correct `g` → `g_corr`; `gk = chunk_cumsum(g_corr) / ln2`; `qg = q * exp2(gk)`; `kbg = k * β * exp2(gk)`; `vb = v * β`; `kg = k * exp2(gk_last - gk)` |
| C0 | cube | `u = A @ vb`; `w = A @ kbg` |

`g_corr` is the safe-gate correction: `lower_bound * sigmoid(exp(A_log)*(g+dt_bias))`. Cumsum always runs on `g_corr`, not raw `g`. `kbg/vb` are workspace. Public tensors are `gk, qg, kg, w, u`.

## Files

- `DESIGN.md`: stage、UB、L1/L0、workspace 槽位
- `INTERFACES.md`: L0 def、L2 aclnn、Torch（`fla_npu.ops.ascendc` / yaml / `torch.ops.npu`）草案
- `kda_gate_wu_golden.py`: CPU V0 / C0
- `run_cpu_golden.py`: CPU runner
- `triton_kernels.py`: staged Triton V0/C0, plus official FLA fused path
- `run_triton.py`: GPU runner
- `run_ct_viz.sh`: `ct viz` CPU=`real`, Triton=`expect`

## Default case

```text
B=1 HK=2 HV=4 T=256 K=128 V=128 chunk=64 bf16 seed=42
use_gate_in_kernel=true safe_gate=true lower_bound=-5
```

CPU compute default is FP32. Triton keeps the generated dtype.

Default case result (CPU FP32 as `real`, staged Triton BF16 as `expect`):

| tensor | stage | abs_mean | abs_max | rel_p99 | cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `g_corr` | V0 | 5.53e-8 | 9.54e-7 | 2.05e-7 | 1.000000 |
| `gk` | V0 | 4.77e-6 | 9.16e-5 | 1.85e-7 | 1.000001 |
| `qg` | V0 | 3.21e-6 | 1.92e-3 | 2.59e-3 | 1.000000 |
| `kbg` | V0 | 1.55e-6 | 1.93e-3 | 2.52e-3 | 1.000000 |
| `vb` | V0 | 3.44e-4 | 1.95e-3 | 3.64e-3 | 1.000002 |
| `kg` | V0 | 2.88e-6 | 1.93e-3 | 2.60e-3 | 1.000001 |
| `w` | C0 | 5.45e-5 | 4.03e-3 | 2.19e-2 | 1.000007 |
| `u` | C0 | 1.98e-3 | 2.04e-2 | 9.53e-2 | 0.9999985 |

`g_corr` is FP32 safe-gate, so it matches CPU almost exactly. Other V0 tensors look like BF16 rounding. `vb/u` do not depend on gate, so they match the previous non-safe-gate run. Staged Triton `gk/qg/u` match official FLA bitwise; `w` differs slightly because staged C0 reloads `kbg` from BF16 workspace. FLA does not expose `g_corr`.

## Environment

```bash
python3 -m venv ~/.venvs/fla
~/.venvs/fla/bin/python3 -m pip install -U pip
~/.venvs/fla/bin/python3 -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
~/.venvs/fla/bin/python3 -m pip install matplotlib numpy einops
~/.venvs/fla/bin/python3 -m pip install https://gitcode.com/Wei_NaChuan/ct/releases/download/v0.9.1/ct_tool-0.9.1-py3-none-any.whl
mkdir -p ~/.local/bin
ln -sfn ~/.venvs/fla/bin/ct ~/.local/bin/ct
git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git ~/golden/fla
```

Python: `~/.venvs/fla/bin/python3`. CT: `~/.local/bin/ct` or the venv `ct`. FLA source: `~/golden/fla`.

## Run

```bash
cd ~/golden/kda_gate_wu_fusion
~/.venvs/fla/bin/python3 run_cpu_golden.py --compute-precision fp32 --out-dir outputs/cpu_fp32
PYTHONPATH=$HOME/golden/fla:${PYTHONPATH} ~/.venvs/fla/bin/python3 run_triton.py --backend staged --out-dir outputs/triton_staged
PYTHONPATH=$HOME/golden/fla:${PYTHONPATH} ~/.venvs/fla/bin/python3 run_triton.py --backend fla --out-dir outputs/triton_fla
CT=$HOME/.venvs/fla/bin/ct MPLBACKEND=Agg bash run_ct_viz.sh outputs/cpu_fp32 outputs/triton_staged outputs/ct_staged
CT=$HOME/.venvs/fla/bin/ct MPLBACKEND=Agg bash run_ct_viz.sh outputs/cpu_fp32 outputs/triton_fla outputs/ct_fla gk qg kg w u
```
