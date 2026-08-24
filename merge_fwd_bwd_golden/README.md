# `merge_fwd_bwd` stage golden

Local debug only. Do not copy machine names, accounts, or absolute host paths into public PRs.

This is **not** the `kda_gate_chunk_cumsum + recompute_w_u` fusion. It is FLA Context Parallel / intracard split: merge per-rank affine state

```text
h ← M @ h
h ← h + He
```

`He` is `[K, V]`, `M` is `[K, K]`. FWD walks past ranks; BWD walks future ranks in reverse. Same pair of stages. `K=128` makes `M @ h` a cube GEMM. The next cube needs this pair’s `h`, so adjacent MM cannot share one AIC stage. `+ He` is a separate vector stage, not cube bias.

## Stages

**No `h0` (CP default, start `h=0`)**

| Stage | Unit | Formula |
| --- | --- | --- |
| V0 | vector | `h = He_0` (`M_0 @ 0` is skipped) |
| C_r | cube | `h = M_r @ h` |
| V_r | vector | `h = h + He_r` |

C_r / V_r repeat for `r = 1 .. R-1`. Default `R=4` is V0, C1, V1, C2, V2, C3, V3.

**With `h0` (intracard)**

| Stage | Unit | Formula |
| --- | --- | --- |
| C_r | cube | `h = M_r @ h` |
| V_r | vector | `h = h + He_r` |

Repeat for `r = 0 .. R-1`. First pair is C0 then V0.

When `state_v_first`, h is `[V, K]`: MM is `h @ M^T`, add is `h + He^T`.

## Files

- `DESIGN.md`: stage、UB、L1/L0、workspace
- `INTERFACES.md`: L0 def、L2 aclnn、Torch（`fla_npu.ops.ascendc` / yaml / `torch.ops.npu`）草案
- `merge_golden.py`: CPU V0 / C_r / V_r
- `triton_kernels.py`: staged Triton (one launch per stage) plus official FLA CP merge
- `run_cpu_golden.py` / `run_triton.py`: runners (`--backend staged|fla`, `--direction fwd|bwd`, `--has-h0`)
- `compare_stats.py`: CPU=`real`, Triton=`expect`
- `run_ct_viz.sh`: `ct viz` with `-sc 100000 -wl 1`
- `run_all.sh`: default fwd / bwd / fwd+h0

FLA CP `merge_fwd_bwd_kernel` does not take `h0`. BWD is implemented by flipping `He/M` on dim 0 and calling `FORWARD=True`. Official FLA still fuses `M @ h + He` in one kernel; only final `h` is compared.

## Default case

```text
R=4 HV=4 K=128 V=128 bf16 seed=42
direction=fwd has_h0=false state_v_first=false
```

CPU compute is FP32. Triton stages compute in FP32, matching FLA.

## Results

CPU FP32 as `real`, staged Triton as `expect`. Cosine stays 1.0; `ct viz` plots sit on `y=x`.

First cube and the following vector add are bitwise equal. Later MM picks up Triton vs CPU FP32 GEMM noise, then the recurrence amplifies it. Staged final `h` matches official FLA (FLA still fuses `M @ h + He` in one kernel).

### fwd, no h0 (V0, C1, V1, C2, V2, C3, V3)

| tensor | stage | abs_mean | abs_max | rel_p99 | cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `h_v0` | V0 | 0 | 0 | 0 | 1.000057 |
| `h_c1` | C1 | 0 | 0 | 0 | 1.000001 |
| `h_v1` | V1 | 0 | 0 | 0 | 1.000000 |
| `h_c2` | C2 | 8.07e-3 | 4.47e-2 | 1.37e-2 | 1.000000 |
| `h_v2` | V2 | 8.07e-3 | 4.47e-2 | 1.38e-2 | 1.000000 |
| `h_c3` / `h_v3` / `h` | C3+V3 | 9.90e-2 | 5.82e-1 | 1.90e-2 | 1.000000 |

Official FLA final `h` vs CPU: abs_mean `9.90e-2`.

### bwd, no h0 (ranks reversed)

| tensor | stage | abs_mean | abs_max | rel_p99 | cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `h_v0` | V0 | 0 | 0 | 0 | 1.000058 |
| `h_c1` | C1 | 0 | 0 | 0 | 1.000000 |
| `h_v1` | V1 | 0 | 0 | 0 | 1.000000 |
| `h_c2` | C2 | 8.09e-3 | 4.60e-2 | 1.38e-2 | 1.000001 |
| `h_v2` | V2 | 8.09e-3 | 4.60e-2 | 1.32e-2 | 1.000001 |
| `h_c3` / `h_v3` / `h` | C3+V3 | 9.88e-2 | 5.99e-1 | 2.00e-2 | 1.000000 |

### fwd, with h0 (C0, V0, … C3, V3)

| tensor | stage | abs_mean | abs_max | rel_p99 | cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `h_c0` | C0 | 0 | 0 | 0 | 1.000001 |
| `h_v0` | V0 | 0 | 0 | 0 | 1.000000 |
| `h_c1` | C1 | 8.19e-3 | 4.94e-2 | 1.36e-2 | 1.000000 |
| `h_v1` | V1 | 8.19e-3 | 4.94e-2 | 1.42e-2 | 1.000001 |
| `h_c2` | C2 | 9.92e-2 | 7.39e-1 | 1.98e-2 | 1.000000 |
| `h_v2` | V2 | 9.92e-2 | 7.39e-1 | 2.03e-2 | 1.000000 |
| `h_c3` / `h_v3` / `h` | C3+V3 | 9.54e-1 | 6.04e+0 | 2.31e-2 | 1.000000 |

No FLA cross-check: CP `merge_fwd_bwd_kernel` does not take `h0`.

## Run

```bash
cd ~/golden/merge_fwd_bwd
bash run_all.sh
```

BWD: add `--direction bwd`. Intracard: add `--has-h0` (staged only).
