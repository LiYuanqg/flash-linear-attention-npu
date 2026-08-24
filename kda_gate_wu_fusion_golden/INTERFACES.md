# `KdaRecomputeWUFwd` 接口（草案）

未落地、未改已发布 ABI。`aclnnKdaGateCumsum`、`aclnnRecomputeWUFwd` 保持原样。本融合是新的私有 L0 + 新 L2，实施前需 `@weinachuan` 确认。

内部布局：dense `BNSD`，varlen `NTD`。`layout` 只出现在 Python / L2，由 L2 转成 L0 内部布局。`kbg/vb`、tile、core 数、workspace offset 只走 tiling data / user workspace，不进 L0 原型。`g` 修正固定 safe gate，没有 `safe_gate`。

`use_exp2=true`（默认）走 `exp2` 与 `/ln2`；`false` 走自然指数 `exp`，cumsum 不再除 `ln2`。safe gate 里的 `exp(A_log)` 不受此开关影响。

```text
g_corr = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   # use_gate_in_kernel=false 时 g_corr = g

use_exp2=true:
    gk  = chunk_cumsum(g_corr) / ln2
    qg  = q * exp2(gk)
    kbg = k * β * exp2(gk)
    kg  = k * exp2(gk_last - gk)

use_exp2=false:
    gk  = chunk_cumsum(g_corr)
    qg  = q * exp(gk)
    kbg = k * β * exp(gk)
    kg  = k * exp(gk_last - gk)

vb = v * β
u  = A @ vb
w  = A @ kbg
```

三层入口：

| 层 | 名字 |
| --- | --- |
| Python | `fla_npu.ops.ascendc.kda_recompute_w_u_fwd` |
| L2 | `aclnnKdaRecomputeWUFwdGetWorkspaceSize` / `aclnnKdaRecomputeWUFwd` |
| L0 / def | `KdaRecomputeWUFwd` |

---

## 1. `fla_npu.ops.ascendc.kda_recompute_w_u_fwd`

稳定 ctypes 入口，不注册 `torch.ops.npu`。`__init__.py` 同时导出 `kda_recompute_w_u_fwd` 与 `npu_kda_recompute_w_u_fwd`。

```python
from fla_npu.ops.ascendc import kda_recompute_w_u_fwd

gk, w, u, qg, kg = kda_recompute_w_u_fwd(
    q, k, v, g, beta, A, chunk_size,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    layout="BSND",
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
)
```

```python
def kda_recompute_w_u_fwd(
    q, k, v, g, beta, A, chunk_size, *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    layout="BSND",
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
) -> tuple:  # (gk, w, u, qg, kg)
    ...
```

ctypes 实参顺序必须与下面 GetWorkspaceSize 去掉末尾 `workspaceSize`、`executor` 后一致。

### 入参

| 名称 | 必选 | 公开 shape（dense，`layout=BNSD`） | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` / `k` | 是 | `[B,H_k,T,K]`；`BSND` 时 `[B,T,H_k,K]` | FP16/BF16 | |
| `v` | 是 | `[B,H_v,T,V]` | 同 `q` | |
| `g` | 是 | `[B,H_v,T,K]` | FP32 | key-wise raw gate |
| `beta` | 是 | `[B,H_v,T]` | FP16/BF16/FP32 | |
| `A` | 是 | `[B,H_v,T,chunk_size]` | 同 `q` | 已求好的 `Akk` |
| `chunk_size` | 是 | 标量 | `int` | 必须等于 `A.shape[-1]`，常用 64 |
| `A_log` | 否 | `[H_v]` | FP32 | `use_gate_in_kernel=True` 时必选 |
| `dt_bias` | 否 | `[H_v,K]` 或 `[H_v*K]` | FP32 | |
| `cu_seqlens` | 否 | `[N+1]` | INT64 | |
| `chunk_indices` | 否 | canonical 列表 | INT64 | 省略时 L2 生成 |
| `layout` | 否 | — | `str` | `BSND` / `BNSD` / `TND` / `NTD`，默认 `BSND` |
| `use_gate_in_kernel` | 否 | — | `bool` | 默认 `True` |
| `use_exp2` | 否 | — | `bool` | 默认 `True`：`exp2`；`False`：`exp` |
| `lower_bound` | 否 | — | `float` | 默认 `-5.0`，仅 gate 开启时有效 |

输出固定 head-major：dense `BNSD`，varlen `NTD`。

| 名称 | dtype | 公开 shape（dense BNSD） |
| --- | --- | --- |
| `gk` | FP32 | `[B,H_v,T,K]` |
| `w` / `qg` / `kg` | 同 `q` | `[B,H_v,T,K]` |
| `u` | 同 `q` | `[B,H_v,T,V]` |

`g_corr` 不返回。`kbg/vb` 留在 aclnn workspace。

```python
import torch
from fla_npu.ops.ascendc import kda_recompute_w_u_fwd

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 256, 128, 128, 64
q = torch.randn(B, T, H_k, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, T, H_v, V, device="npu", dtype=torch.bfloat16)
g = torch.randn(B, T, H_v, K, device="npu", dtype=torch.float32)
beta = torch.rand(B, T, H_v, device="npu", dtype=torch.float32)
A = torch.randn(B, T, H_v, chunk_size, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
dt_bias = torch.randn(H_v, K, device="npu", dtype=torch.float32)

gk, w, u, qg, kg = kda_recompute_w_u_fwd(
    q, k, v, g, beta, A, chunk_size,
    A_log=A_log,
    dt_bias=dt_bias,
    layout="BSND",
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
)
```

yaml / legacy 同名：`npu_kda_recompute_w_u_fwd`。schema 与 L2 参数对齐，落地时写入 `npu_custom.yaml`。`torch.ops.npu.npu_kda_recompute_w_u_fwd` 仅在 `load_legacy_torch_ops()` 后可用，不是默认路径。

```text
npu_kda_recompute_w_u_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, int chunk_size, *,
    Tensor? A_log=None, Tensor? dt_bias=None,
    int[]? cu_seqlens=None, int[]? chunk_indices=None,
    str layout="BSND",
    bool use_gate_in_kernel=True, bool use_exp2=True, float lower_bound=-5.0
) -> (Tensor gk, Tensor w, Tensor u, Tensor qg, Tensor kg)
```

---

## 2. L2 `aclnnKdaRecomputeWUFwdGetWorkspaceSize`

`op_host/op_api/aclnn_kda_recompute_w_u_fwd.h`

顺序：必选输入 → 可选输入 → 属性 → 输出 → `workspaceSize` → `executor`。

```cpp
aclnnStatus aclnnKdaRecomputeWUFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    int64_t chunkSize,
    bool useGateInKernel,
    bool useExp2,
    double lowerBound,
    const aclTensor *gkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnKdaRecomputeWUFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

L2 职责：

- `layout` ∈ `{BSND,BNSD,TND,NTD}`，转成 L0 内部 `BNSD`/`NTD`
- 校验 `chunkSize == a->GetViewShape()` 最后一维，并写入 L0 `chunk_size`
- 把 `useGateInKernel` / `useExp2` / `lowerBound` 原样下给 L0
- 公开输出写成 head-major（`gk/w/qg/kg/u`）
- 申请 `kbg/vb` user workspace；`sysWorkspaceSize` 另计

ctypes 类型表必须逐项对照本原型，禁止只按相邻算子推测。

---

## 3. L0 / def `KdaRecomputeWUFwd`

`op_host/kda_recompute_w_u_fwd_def.cpp`

没有 `layout`。`chunk_size`、`use_exp2` 是 L0 属性。`H_k/H_v/T/K/V` 仍从 descriptor 读。

```cpp
class KdaRecomputeWUFwd : public OpDef {
public:
    explicit KdaRecomputeWUFwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("A").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("a_log").ParamType(OPTIONAL).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);

        this->Output("gk").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("w").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("u").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("qg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("kg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(true);
        this->Attr("use_exp2").AttrType(REQUIRED).Bool(true);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
    }
};
```

### 输入（内部 layout）

| 名称 | 必选 | dtype | 内部 shape |
| --- | --- | --- | --- |
| `q` | 是 | FP16/BF16 | `[B,H_k,T,K]` |
| `k` | 是 | 同 `q` | `[B,H_k,T,K]` |
| `v` | 是 | 同 `q` | `[B,H_v,T,V]` |
| `g` | 是 | FP32 | `[B,H_v,T,K]` |
| `beta` | 是 | FP16/BF16/FP32 | `[B,H_v,T]` |
| `A` | 是 | 同 `q` | `[B,H_v,T,chunk_size]` |
| `a_log` | 否 | FP32 | `[H_v]` |
| `dt_bias` | 否 | FP32 | `[H_v,K]` |
| `cu_seqlens` | 否 | INT64 | `[N+1]` |
| `chunk_indices` | 否 | INT64 | canonical 列表 |

### 输出

| 名称 | 必选 | dtype | 内部 shape |
| --- | --- | --- | --- |
| `gk` | 是 | FP32 | `[B,H_v,T,K]` |
| `w` | 是 | 同 `q` | `[B,H_v,T,K]` |
| `u` | 是 | 同 `q` | `[B,H_v,T,V]` |
| `qg` | 是 | 同 `q` | `[B,H_v,T,K]` |
| `kg` | 是 | 同 `q` | `[B,H_v,T,K]` |

不进原型：`g_corr`（UB）、`kbg/vb`（user workspace）。

### 属性

| 名称 | 默认 | 说明 |
| --- | --- | --- |
| `chunk_size` | `64` | chunk 长度；L2 保证等于 `A` 最后一维 |
| `use_gate_in_kernel` | `true` | `true`：safe gate；`false`：`g_corr=g`，cumsum 仍做 |
| `use_exp2` | `true` | `true`：`gk/=ln2` 且 `exp2`；`false`：`gk` 不除 `ln2`，用 `exp` |
| `lower_bound` | `-5.0` | 仅 `use_gate_in_kernel=true` 时有效 |

没有 `safe_gate`、`layout`、`logical_*`、`stage`。
