# `ChunkKdaBwdRecompute` 接口（修订版）

Recompute 仅在 disable_recompute=False 分支执行，重新生成反向所需的 gk/w/u/qg/kg；disable_recompute=True 时由外层 L2 直接绑定正向保存缓存并跳过本 L0。

## 1. `fla_npu.ops.ascendc.chunk_kda_bwd_recompute` 接口定义
```python
from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

gk, w, u, qg, kg = chunk_kda_bwd_recompute(
    q,
    k,
    v,
    g,
    beta,
    A,
    chunk_size,
    *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
)
```
内层接口固定 head-first：dense BNSD [B,H,T,D]，varlen NTD [H,T,D]，不做 permute。 A 即正向 Akk。safe gate 固定为：
```text
g_corr = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))

use_exp2=True:
    gk = chunk_cumsum(g_corr) / ln2
    qg = q * exp2(gk)
    kbg = k * beta * exp2(gk)
    kg = k * exp2(gk_last - gk)

use_exp2=False:
    gk = chunk_cumsum(g_corr)
    qg = q * exp(gk)
    kbg = k * beta * exp(gk)
    kg = k * exp(gk_last - gk)

vb = v * beta
u = A @ vb
w = A @ kbg
```

| 输出 | dense BNSD | varlen NTD | dtype |
| --- | ----| --- | ---|
| gk | [B,HV,T,K] | [HV,T,K] | FP32 |
| w/qg/kg | [B,HV,T,K] | [HV,T,K] | q/k dtype |
| u | [B,HV,T,V] | [HV,T,V] | v dtype|


g_corr 不返回，kbg/vb 只存在于 executor workspace。 本 L0 的 g/beta 支持 BF16、FP32，不支持 FP16；A_log/dt_bias 仅支持FP32， gate 变换使用 FP32 计算。
## 2. aclnn 接口定义
```python
ACLNN_API aclnnStatus aclnnChunkKdaBwdRecomputeGetWorkspaceSize(
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
    int64_t chunkSize,
    // bool useGateInKernel, //去掉，通过gkOutOptional==null判断
    bool useExp2,
    double lowerBound,
    const aclTensor *gkOutOptional, //当useGateInKernel=false，直接传nullptr
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnChunkKdaBwdRecompute(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```
私有 L0 原型：
```python
namespace l0op {
const std::array<const aclTensor *, 5> ChunkKdaBwdRecompute(
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
    int64_t chunkSize,
    bool useGateInKernel,
    bool useExp2,
    double lowerBound,
    const aclTensor *gkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    aclOpExecutor *executor);
}
```
## 3. Def 接口定义
```python
OpDef: ChunkKdaBwdRecompute

Inputs:
  q, k, v, g, beta, A               REQUIRED
  a_log, dt_bias                     OPTIONAL
  cu_seqlens, chunk_indices          OPTIONAL, value-dependent INT64 Tensor

Attrs:
  chunk_size                         REQUIRED int = 64
  use_gate_in_kernel                 REQUIRED bool = true
  use_exp2                           REQUIRED bool = true
  lower_bound                        OPTIONAL float = -5.0

Outputs:
  gk, w, u, qg, kg                  REQUIRED
```