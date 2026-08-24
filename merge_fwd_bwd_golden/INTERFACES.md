# `merge_fwd_bwd`：L0 / L2 / Torch 接口（草案）

未落地。FLA CP 的 `merge_fwd_bwd_kernel` 没有对应的已发布 `aclnn*`。本设计是新的私有 L0 + 新 L2，实施前需 `@weinachuan` 确认。

`R`、`H_v`、`K`、`V` 从 tensor descriptor 读，不做 L0 属性。核数、`BV`、L1 常驻 `h` 只走 tiling data。

## L0：`MergeFwdBwd`

`op_host/merge_fwd_bwd_def.cpp`

```cpp
class MergeFwdBwd : public OpDef {
public:
    explicit MergeFwdBwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("he").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("m").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("h0").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("h").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("reverse").AttrType(REQUIRED).Bool(false);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);
    }
};
```

`he` 与 `m` 分开传，不把 FLA 的 `[He | M]` pack 做成 L0 入参。L2 若拿到 packed `ag_hm`，切开再下 L0。

### 输入

| 名称 | 必选 | dtype | shape | 语义 |
| --- | --- | --- | --- | --- |
| `he` | 是 | FP16/BF16/FP32 | `[R,H_v,K,V]` | 各 rank 加性状态 |
| `m` | 是 | 同 `he` | `[R,H_v,K,K]` | 各 rank 线性映射 |
| `h0` | 否 | FP32 | `[H_v,K,V]` 或 `[H_v,V,K]` | 卡内起点；缺省视为 `h=0`，跳过 `M_0@0` |

`he[r]` / `m[r]` 的 `r` 沿 dim0。FWD：`0 .. R-1`。BWD：`R-1 .. 0`（由 `reverse` 决定，不在 host 上翻数据）。

### 输出

| 名称 | 必选 | dtype | shape |
| --- | --- | --- | --- |
| `h` | 是 | FP32 | `[H_v,K,V]` 或 `[H_v,V,K]` |

与 `state_v_first` 一致。中间 `h_c*` / `h_v*` 不暴露。

### 属性

| 名称 | 默认 | 说明 |
| --- | --- | --- |
| `reverse` | `false` | `false`：FWD，过去 rank；`true`：BWD，未来 rank 倒序 |
| `state_v_first` | `false` | `true` 时 `h/h0` 为 `[V,K]`，更新 `h@M^T`、`h+He^T` |

没有 `num_ranks`、`rank`、`forward` 第二份布尔（与 `reverse` 重复）、`BV`。

`h0==nullptr` 与 `h0` 非空走同一 L0；tiling 设 `HAS_H0`。

## L2：`aclnnMergeFwdBwd`

```cpp
aclnnStatus aclnnMergeFwdBwdGetWorkspaceSize(
    const aclTensor *he,
    const aclTensor *m,
    const aclTensor *h0Optional,
    bool reverse,
    bool stateVFirst,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnMergeFwdBwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

L2 校验：`he` 与 `m` 的 `R,H_v,K` 一致；`m` 末两维相等且等于 `he` 的 K；`h0` 若有则与 `hOut` layout 一致；`hOut` 为 FP32。

若调用方只有 packed `ag_hm` `[R,H_v,K,V+K]`，L2 切成 view 再下 L0，不把 pack 写进 L0。

可选重载（不进 L0）：

```cpp
aclnnStatus aclnnMergeFwdBwdFromPackedGetWorkspaceSize(
    const aclTensor *agHm,
    const aclTensor *h0Optional,
    bool reverse,
    bool stateVFirst,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

默认只提供分开的 `he/m` 这一条 L2。packed 重载是否加，同样要 `@weinachuan` 确认。

## Torch

三条通路表达同一语义。默认新代码走 `fla_npu.ops.ascendc` ctypes。yaml / `torch.ops.npu` 是 legacy 兼容。

| 通路 | 入口 | 何时用 |
| --- | --- | --- |
| 稳定 Python | `fla_npu.ops.ascendc.merge_fwd_bwd` | 默认 |
| 稳定 Python 别名 | `fla_npu.ops.ascendc.npu_merge_fwd_bwd` | 与 yaml 同名 |
| aclnn | `aclnnMergeFwdBwdGetWorkspaceSize` / `aclnnMergeFwdBwd` | C++ / ctypes |
| legacy | `torch.ops.npu.npu_merge_fwd_bwd` | 显式 `fla_npu.load_legacy_torch_ops()` |

没有 `torch.autograd.Function`。`R/H_v/K/V` 从 tensor 读，不进属性。

| 名称 | 公开 shape | dtype |
| --- | --- | --- |
| `he` | `[R,H_v,K,V]` | FP16/BF16/FP32 |
| `m` | `[R,H_v,K,K]` | 同 `he` |
| `h0` | `[H_v,K,V]`；`state_v_first=True` 时 `[H_v,V,K]` | FP32 |
| `h` | 与 `h0` 相同 layout | FP32 |

### 稳定入口 `fla_npu.ops.ascendc`

`__init__.py` 同时导出 `merge_fwd_bwd` 与 `npu_merge_fwd_bwd`。

```python
import torch
from fla_npu.ops.ascendc import merge_fwd_bwd

h = merge_fwd_bwd(
    he, m,
    h0=None,
    reverse=False,
    state_v_first=False,
)
```

ctypes wrapper：

```python
def npu_merge_fwd_bwd(
    he, m, *,
    h0=None,
    reverse=False,
    state_v_first=False,
):
    ...
```

实参顺序与 `aclnnMergeFwdBwdGetWorkspaceSize` 去掉 `workspaceSize`、`executor` 后一致。Wrapper 分配 FP32 的 `h`；`h0 is None` 时走无 h0 路径（V0 拷贝 `He_0`）。若调用方只有 packed `ag_hm`，在 Python 里切成 `he/m` 再调，不要把 pack 写进 L0。

调用示例（默认 case，无 h0、FWD、R=4）：

```python
import torch
from fla_npu.ops.ascendc import merge_fwd_bwd

R, H_v, K, V = 4, 4, 128, 128
device, dtype = "npu", torch.bfloat16
he = torch.randn(R, H_v, K, V, device=device, dtype=dtype)
m = torch.randn(R, H_v, K, K, device=device, dtype=dtype)

h = merge_fwd_bwd(he, m, reverse=False, state_v_first=False)
# h: [H_v, K, V] float32
```

有 `h0`：

```python
h0 = torch.zeros(H_v, K, V, device=device, dtype=torch.float32)
h = merge_fwd_bwd(he, m, h0=h0, reverse=False, state_v_first=False)
```

BWD：`reverse=True`。`state_v_first=True` 时 `h0/h` 为 `[H_v,V,K]`。

### `npu_custom.yaml` / native schema

```text
npu_merge_fwd_bwd(
    Tensor he, Tensor m, *,
    Tensor? h0=None, bool reverse=False, bool state_v_first=False
) -> Tensor h
```

schema、`FLANpuOpApi.cpp`、ctypes 类型表、L2 原型必须同一变更对齐。

### legacy `torch.ops.npu`

```cpp
at::Tensor npu_merge_fwd_bwd(
    const at::Tensor &he,
    const at::Tensor &m,
    const c10::optional<at::Tensor> &h0,
    bool reverse,
    bool state_v_first);
```

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()
h = torch.ops.npu.npu_merge_fwd_bwd(
    he, m,
    h0=None,
    reverse=False,
    state_v_first=False,
)
```

未 `load_legacy_torch_ops()` 时该名字不存在。新测试和示例默认用 `fla_npu.ops.ascendc.merge_fwd_bwd`。
