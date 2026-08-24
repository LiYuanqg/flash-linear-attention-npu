# KDA Gate+WU 融合：L0 / L2 / Torch 接口（草案）

未落地、未改已发布 ABI。`aclnnKdaGateCumsum`、`aclnnRecomputeWUFwd` 保持原样。本融合是新的私有 L0 + 新 L2，实施前需 `@weinachuan` 确认。

内部布局与其它 KDA 算子一致：dense 用 BNSD，varlen 用 NTD。L2 负责公开 layout 转内部布局。`kbg/vb`、tile、core 数、workspace offset 只走 tiling data / user workspace，不进 L0 原型。

`g` 修正固定 safe gate，L0 **没有** `safe_gate` 属性。

## L0：`KdaRecomputeWUFwd`

`op_host/kda_recompute_w_u_fwd_def.cpp`

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

        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(true);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
    }
};
```

### 输入

| 名称 | 必选 | dtype | 内部 shape | 语义 |
| --- | --- | --- | --- | --- |
| `q` | 是 | FP16/BF16 | `[B,H_k,T,K]` | |
| `k` | 是 | 同 `q` | `[B,H_k,T,K]` | |
| `v` | 是 | 同 `q` | `[B,H_v,T,V]` | |
| `g` | 是 | FP32 | `[B,H_v,T,K]` | key-wise raw gate |
| `beta` | 是 | FP16/BF16/FP32 | `[B,H_v,T]` | |
| `A` | 是 | 同 `q` | `[B,H_v,T,BT]` | 已求好的 `Akk` |
| `a_log` | 否 | FP32 | `[H_v]` | safe gate 用 |
| `dt_bias` | 否 | FP32 | `[H_v,K]` | |
| `cu_seqlens` | 否 | INT64 | `[N+1]` | |
| `chunk_indices` | 否 | INT64 | canonical 列表 | 省略时 L2 生成 |

`chunk_size` = `A` 最后一维，`H_k/H_v/T/K/V` 从 descriptor 读，都不做 L0 属性。

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
| `use_gate_in_kernel` | `true` | `true`：safe gate `lower_bound * sigmoid(exp(A_log)*(g+dt_bias))`；`false`：`g_corr=g`，cumsum 仍做 |
| `lower_bound` | `-5.0` | 仅 `use_gate_in_kernel=true` 时有效 |

没有 `safe_gate`、`layout`、`logical_*`、`stage`。

## L2：`aclnnKdaRecomputeWUFwd`

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

L2 职责：校验 `chunkSize == A.shape[-1]`；`layout` ∈ `{BSND,BNSD,TND,NTD}` 并转成 L0 内部布局；公开输出转回调用方 layout（`gk/w/qg/kg/u` 与现有 KDA 反向中间量一致，head-major）；申请 `kbg/vb` workspace。

GetWorkspaceSize 参数顺序：必选输入 → 可选输入 → 属性 → 输出 → `workspaceSize` → `executor`。ctypes 表必须逐项对照。

## Torch

三条通路表达同一语义。默认新代码走 `fla_npu.ops.ascendc` ctypes，不注册 `torch.ops.npu`。yaml / `torch.ops.npu` 是 legacy 兼容，落地时与 L2 原型同步，但不要作为唯一调用方式。

| 通路 | 入口 | 何时用 |
| --- | --- | --- |
| 稳定 Python | `fla_npu.ops.ascendc.kda_recompute_w_u_fwd` | 默认 |
| 稳定 Python 别名 | `fla_npu.ops.ascendc.npu_kda_recompute_w_u_fwd` | 与 yaml 同名 |
| aclnn | `aclnnKdaRecomputeWUFwdGetWorkspaceSize` / `aclnnKdaRecomputeWUFwd` | C++ / ctypes |
| legacy | `torch.ops.npu.npu_kda_recompute_w_u_fwd` | 显式 `fla_npu.load_legacy_torch_ops()` |

没有 `torch.autograd.Function`：这是反向重计算 helper，输出不接 autograd。

公开 `layout` 只解释 `q/k/v/g/beta/A` 输入。输出与现有 KDA 反向中间量一致，固定 **head-major**：dense `BNSD`，varlen `NTD`。L2 把调用方 layout 转成 L0 内部布局后再转回这套输出布局。

| 名称 | 公开 shape（dense BNSD） | dtype |
| --- | --- | --- |
| `q` / `k` | `[B,H_k,T,K]`；`layout=BSND` 时入参为 `[B,T,H_k,K]` | FP16/BF16 |
| `v` | `[B,H_v,T,V]` | 同 `q` |
| `g` | `[B,H_v,T,K]` | FP32 |
| `beta` | `[B,H_v,T]` | FP16/BF16/FP32 |
| `A` | `[B,H_v,T,BT]` | 同 `q` |
| `A_log` | `[H_v]` | FP32 |
| `dt_bias` | `[H_v,K]` 或 `[H_v*K]` | FP32 |
| `gk` | `[B,H_v,T,K]` | FP32 |
| `w` / `qg` / `kg` | `[B,H_v,T,K]` | 同 `q` |
| `u` | `[B,H_v,T,V]` | 同 `q` |

`chunk_size` 必须等于 `A.shape[-1]`。`use_gate_in_kernel=True` 时 `A_log` 必选。没有 `safe_gate` 参数。

### 稳定入口 `fla_npu.ops.ascendc`

`fla_npu/ops/ascendc/__init__.py` 同时导出去前缀名和 `npu_` 名，二者指向同一 ctypes wrapper。

```python
import torch
from fla_npu.ops.ascendc import kda_recompute_w_u_fwd

gk, w, u, qg, kg = kda_recompute_w_u_fwd(
    q, k, v, g, beta, A,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    layout="BSND",
    chunk_size=64,
    use_gate_in_kernel=True,
    lower_bound=-5.0,
)
```

ctypes 实参顺序必须与 `aclnnKdaRecomputeWUFwdGetWorkspaceSize` 去掉末尾 `workspaceSize`、`executor` 后一致；类型表写在 `_aclnn_ctypes.py`，逐项对照，禁止只按相邻算子推测。

```python
def npu_kda_recompute_w_u_fwd(
    q, k, v, g, beta, A, *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    layout="BSND",
    chunk_size=64,
    use_gate_in_kernel=True,
    lower_bound=-5.0,
):
    ...
```

Wrapper 负责：校验 layout / `chunk_size == A.shape[-1]`；按 layout 推 `B,H_k,H_v,T,K,V`；分配 `gk`(FP32)、`w/u/qg/kg`（与 `q` 同 dtype，head-major）；把 `kbg/vb` 留在 aclnn workspace，不暴露给调用方。

调用示例（默认 case）：

```python
import torch
from fla_npu.ops.ascendc import kda_recompute_w_u_fwd

B, H_k, H_v, T, K, V, BT = 1, 2, 4, 256, 128, 128, 64
device, dtype = "npu", torch.bfloat16
q = torch.randn(B, T, H_k, K, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn(B, T, H_v, V, device=device, dtype=dtype)
g = torch.randn(B, T, H_v, K, device=device, dtype=torch.float32)
beta = torch.rand(B, T, H_v, device=device, dtype=torch.float32)
A = torch.randn(B, T, H_v, BT, device=device, dtype=dtype)
A_log = torch.randn(H_v, device=device, dtype=torch.float32)
dt_bias = torch.randn(H_v, K, device=device, dtype=torch.float32)

gk, w, u, qg, kg = kda_recompute_w_u_fwd(
    q, k, v, g, beta, A,
    A_log=A_log,
    dt_bias=dt_bias,
    layout="BSND",
    chunk_size=BT,
    use_gate_in_kernel=True,
    lower_bound=-5.0,
)
# gk: [B, H_v, T, K] float32
# w/qg/kg: [B, H_v, T, K] bf16
# u: [B, H_v, T, V] bf16
```

### `npu_custom.yaml` / native schema

写入 `torch_custom/fla_npu/npu_custom.yaml`（以及生成链用的 `test_native_functions.yaml`）：

```text
npu_kda_recompute_w_u_fwd(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, *,
    Tensor? A_log=None, Tensor? dt_bias=None,
    int[]? cu_seqlens=None, int[]? chunk_indices=None,
    str layout="BSND", int chunk_size=64,
    bool use_gate_in_kernel=True, float lower_bound=-5.0
) -> (Tensor gk, Tensor w, Tensor u, Tensor qg, Tensor kg)
```

schema、`FLANpuOpApi.cpp` 包装、ctypes 类型表、本 L2 原型必须同一变更对齐。该 yaml 只服务 legacy `torch.ops.npu`，不改变稳定 ctypes 路径。

### legacy `torch.ops.npu`

`op_plugin/ops/opapi/FLANpuOpApi.cpp` 草案：

```cpp
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_kda_recompute_w_u_fwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &beta,
    const at::Tensor &A,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    at::OptionalIntArrayRef cu_seqlens,
    at::OptionalIntArrayRef chunk_indices,
    c10::string_view layout,
    int64_t chunk_size,
    bool use_gate_in_kernel,
    double lower_bound);
```

包装层分配五份输出后 `EXEC_NPU_CMD_EXT(aclnnKdaRecomputeWUFwd, ...)`。参数顺序与 yaml / GetWorkspaceSize 一致。

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()
gk, w, u, qg, kg = torch.ops.npu.npu_kda_recompute_w_u_fwd(
    q, k, v, g, beta, A,
    A_log=A_log,
    dt_bias=dt_bias,
    layout="BSND",
    chunk_size=64,
    use_gate_in_kernel=True,
    lower_bound=-5.0,
)
```

未 `load_legacy_torch_ops()` 时该名字不存在。新测试和示例默认用 `fla_npu.ops.ascendc.kda_recompute_w_u_fwd`。
