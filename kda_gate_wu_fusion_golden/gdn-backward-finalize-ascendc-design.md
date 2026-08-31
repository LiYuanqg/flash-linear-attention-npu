# GDN backward finalize 全融合 Ascend C 算子设计

## 1. 目标

本文设计一个覆盖 stage golden 全部计算的 GDN backward `finalize` 全融合算子。
算子完整包含：

```text
chunk_bwd_dqkwg
  -> prepare_wy_repr_bwd
  -> dk/dg 合成
  -> chunk 内 reverse cumsum
  -> Q/K L2Norm backward
  -> beta sigmoid backward
  -> GDN gate backward
```

数学边界严格对齐：

- `gdn_backward_golden.py`：Stage 0--9 逐 stage 公式；
- `run_split_stage_golden.py`：完整调用和最终输出；
- `run_triton_dqkwg_prepare.py`：Triton 真实调用链。

实现结构参考 Ascend C
[`chunk_gated_delta_rule_bwd_dhu`](https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu)
的混合 AIC/AIV、定长/变长 offset helper、跨核同步和 workspace 管理方式，但采用不同的
逻辑分核：**只按 chunk 分核，不按 head 分核**。每个 chunk task 在核内遍历所有
`HK/HV` head。本版本仅支持 GVA 比例 `1:1`--`1:4`，即 `G=HV/HK in {1,2,3,4}`。
核内使用动态 HV 任务组：`G=3` 时组大小取 3，`G=1/2/4` 时取 4。对每个
任务组，两个 AIV 按任务序号交错承包完整 HV：AIV0 处理 0、2、4...，AIV1
处理 1、3、5...，不切分单个 HV 内部数据。两个 AIV 不设置 stage 级汇合；每个
AIV 完成自己承包的当前 stage 任务后即可独立进入下一 stage。只有存在真实的跨
AIV 数据依赖或归约时才增加对应同步。执行方式参考 `bwd_dhu` 的 head round，但
不固定为 4 个 HV。

本文 Stage 0--12 是按 L1/UB 驻留约束重新规划后的目标编号，不沿用当前代码中的旧
Stage 编号。后续实现应整体按本文的公式、数据流、workspace/UB 布局和 AIV 交错
分工重新对应。

## 2. 范围与 shape

第一版目标：

```text
SoC               Ascend 950 only
q/k layout        [B, HK, T, K]
value/gate layout [B, HV, T]
state layout      [B, HV, NT, K, V]
K                 128
V                 128
chunk_size        64
dtype             bf16，gate/beta 允许 fp32
GVA               HV % HK == 0, G=HV/HK in {1,2,3,4}
```

目标 case：

```text
B=1, HK=16, HV=32, T=12288, K=V=128, BT=64, BF16
```

`BT/K/V` 是固定规格：`BT=64`、`K=128`、`V=128`。host 侧必须明确拦截其它值，
不能选择其它 tiling 分支或静默执行。尾 chunk 和 varlen 仍允许当前有效长度 `M<64`。
算子定义和编译配置只注册 `ascend950`，不提供 910B、910_93 或其它 SoC 的 fallback
kernel。

本算子不包含前序 `bwd_dhu` 隐藏状态递推，而是接收已经生成的 `h/dh`。golden
中生成但未参与 Stage 0--9 公式的 `_w` 不进入接口。

符号：

```text
B   batch size
T   定长每 batch token 数；varlen 时为 packed token 总数
HK  q/k head 数
HV  value/gate head 数
G   HV / HK
K   q/k dim
V   value dim
BT  chunk_size
NT  总 chunk 数
M   当前 chunk 有效 token 数，M <= BT
TG  HV task group 大小，`TG = (G == 3) ? 3 : 4`
hk  当前 hv 映射的 q/k head，hk = hv / G
```

主要张量：

| 张量 | Shape | 说明 |
|---|---:|---|
| `q, k, dq, dk` | `[B,HK,T,K]` | q/k 已是 L2Norm forward 输出 |
| `v, v_new, do, du, dv` | `[B,HV,T,V]` | value 侧输入和输出 |
| `g, beta, dbeta, dg` | `[B,HV,T]` | `g/beta` 为变换后值 |
| `g_input, beta_raw` | `[B,HV,T]` | gate/sigmoid 变换前值 |
| `h, dh` | `[B,HV,NT,K,V]` | 每 chunk state，head-first |
| `A` | `[B,HV,T,BT]` | 每 token 行存一个 chunk 内矩阵行 |
| `q_rstd, k_rstd` | `[B,HK,T]` | L2Norm forward 保存值，fp32 |
| `A_log, dt_bias` | `[HV]` | gate 参数，`dt_bias` 可选 |

golden 单 chunk 中将 `h/dh` 读取为逻辑 `[V,K]`，将 `A` 读取为逻辑
`[BT,BT]`。kernel 的 GM layout 和 Cube 搬运必须保持该转置语义。

## 3. 算子接口

接口名称固定为：

```text
GE/Ascend C 算子名：ChunkGdnBwdFinalize
ACLNN workspace：  aclnnChunkGdnBwdFinalizeGetWorkspaceSize
ACLNN execute：    aclnnChunkGdnBwdFinalize
Python：           fla_npu.ops.ascendc.chunk_gdn_bwd_finalize
```

### 3.1 输入

| 名称 | Shape | dtype | 来源/用途 |
|---|---:|---|---|
| `q` | `[B,HK,T,K]` | bf16 | 归一化 Q |
| `k` | `[B,HK,T,K]` | bf16 | 归一化 K |
| `v` | `[B,HV,T,V]` | 同 `q` | prepare backward |
| `v_new` | `[B,HV,T,V]` | 同 `q` | dqkwg value 输入 |
| `dox` | `[B,HV,T,V]` | 同 `q` | 上游输出梯度 |
| `du` | `[B,HV,T,V]` | 同 `q` | 原始 value 梯度 |
| `g` | `[B,HV,T]` | bf16/fp32 | 变换后的 chunk gate |
| `beta` | `[B,HV,T]` | 同 `g` | sigmoid 后 beta |
| `h` | `[B,HV,NT,K,V]` | 同 `q` | forward chunk state |
| `dh` | `[B,HV,NT,K,V]` | 同 `q` | `bwd_dhu` 输出 |
| `A` | `[B,HV,T,BT]` | 同 `q` | WY inverse/中间矩阵 |
| `q_rstd` | `[B,HK,T]` | fp32 | Q L2Norm 保存值 |
| `k_rstd` | `[B,HK,T]` | fp32 | K L2Norm 保存值 |
| `g_input` | `[B,HV,T]` | 同 `g` | GDN gate 原始输入 |
| `beta_raw` | `[B,HV,T]` | 同 `g` | beta sigmoid 原始输入 |
| `A_log` | `[HV]` | fp32 | GDN gate 参数 |
| `dt_bias` | `[HV]` | fp32 | 可选 gate bias |
| `cu_seqlens` | `[seqNum+1]` | int64 | 可选 varlen 序列边界 |
| `chunk_indices` | `[2*NT]` | int64 | 可选 `(seqIdx,localChunkIdx)` pair |

属性：

| 名称 | 类型 | 默认值 |
|---|---|---:|
| `scale` | float | `1/sqrt(K)` |
| `chunk_size` | int64 | 64 |
| `use_gate_in_kernel` | bool | true |
| `use_qk_l2norm_in_kernel` | bool | true |
| `use_beta_sigmoid_in_kernel` | bool | true |

`du` 必须保持调用本算子前的原始值。dqkwg 只用它生成 `dw0`，不能先覆盖成其它
`dv` 再传入 prepare 路径。

参数约束：

- `scale=None` 时 wrapper 传 `1/sqrt(128)`；
- `chunk_size` 为兼容上层调用保留，但只接受 `64`；
- `use_qk_l2norm_in_kernel=True` 时 `q_rstd/k_rstd` 必须非空；
- `use_beta_sigmoid_in_kernel=True` 时 `beta_raw` 必须非空；
- `use_gate_in_kernel=True` 时 `g_input/A_log` 必须非空；
- `dt_bias` 始终可选；非空时要求 `use_gate_in_kernel=True`；
- `cu_seqlens/chunk_indices` 必须同时为 `None` 或同时非空；
- 返回顺序固定为
  `(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)`；开关关闭或 bias 不存在时，对应
  `dA_log/ddt_bias` 返回 `None`。


## 4. Stage 0--12 完整数学语义

以下按单个 `(chunk,hv)` 描述。令 `hk=hv/G`，当前 chunk 有效长度为 `M`。
本轮只确定 Stage、完整张量搬运和容量，不确定最终计算精度；资源表暂按大型张量
BF16、小向量与归约标量 FP32 估算。下文统一标注
完整 chunk 的逻辑 shape；尾 chunk 仍按相同 `[BT,...]` 逻辑 shape 描述，但只有前
`M` 行有效，矩阵只有左上角 `M x M` 有效。

单个 `(chunk,hv)` 的基础 shape 为：

```text
q, k                       # [BT,K]
v, v_new, do, du           # [BT,V]
g, beta                    # [BT]
h, dh                      # [V,K]
A                          # [BT,BT]
q_rstd, k_rstd             # [BT]
g_input, beta_raw          # [BT]
A_log[hv], dt_bias[hv]     # scalar
```

本节保持 golden Stage 0--9 的数学结果，只把
`dk0 -> dkBase -> dkIntra -> dkHv` 调整到 DQ 公式之后计算。

记当前 K head 对应的 GVA value-head 集合为：

```text
H(hk) = { hv_i | floor(hv_i / G) == hk }
```

除非公式中另有说明，token 下标 `i/j/t` 的有效范围都是 `[0,M)`；`rowSum` 删除
最后一维，`colSum` 删除倒数第二维，`tril` 不改变输入 shape。

本节所有地址均以整块物理存储起点为 `0 KiB`，使用半开区间 `[start,end)`：

```text
L1[0,512)      当前 Cube Stage 可使用的完整 L1 空间
UB[0,248)      当前 Vector Stage 可使用的完整 UB 空间
```

后续“空间布局”中的偏移均指上述绝对 KiB 偏移。L1 和 UB 内部没有固定
输入区、tensor 区或小向量区边界；任一 Stage 可在完整物理空间内
重新安排已结束生命周期的地址。UB 采用两端向中间挤压：大型 BF16 tensor 从
`UB[0)` 向高地址增长，FP32 `[2,BT]` 小向量从 `UB[248)` 按 0.5 KiB 子槽
向低地址增长。该策略只用于 tensor 首次分配；一旦分配，生命周期内物理地址固定。
因此短生命周期 tensor 释放后，后续 Stage 可能在长生命周期 tensor 之间留下内部
空洞，这类空洞只能分配给尺寸匹配的新 tensor，不能通过搬移存活 tensor 消除。
若保存两个 scalar，则对应 0.5 KiB 子槽中
只有开头 8 Byte 有效。这只是当前生命周期布局，不是不可借用的物理分区。
L1 中标记为 `[4]` 的区间
按四等份连续存放 HV0--HV3。例如
`L1[128,256)=h[4]` 等价于 `L1[128,160)=h[HV0]`、
`L1[160,192)=h[HV1]`、`L1[192,224)=h[HV2]`、
`L1[224,256)=h[HV3]`；其余 `[4]` 区间采用相同等分规则。

所有 Stage 都必须实现可验证的双缓冲流水，不允许只在文档中写
“可并发”而不给出物理 slot 和事件闭环。Vector Stage 使用两份
UB ping/pong：当前 HV 的 VF 执行时，MTE2 将下一个交错 HV
搬入另一 slot；当前结果由 MTE3 读取时，V 可继续使用另一
slot。每个 UB slot 使用 `MTE2_V -> V_MTE3 -> MTE3_MTE2`
闭环；覆盖 slot 前必须等上一轮 MTE3 读完，`MTE2_V`
负责保证后续 V 只消费新搬入的数据。Cube Stage 的 L0A、L0B、L0C 分别维护独立 ping/pong，每次
使用后立即取反；同时用不同 resident 的独立 flag 发射
下一 HEAD 的 GM->L1 搬运，使 MTE2 与当前 MTE1/Cube/Fixpipe 重叠。每层流水
都要按 slot 建立生产者/消费者事件，复用前闭环上一轮依赖。

任务组之间不允许仅依据“当前已实现的最后一个 Stage 完成”就复用仍属于完整
S0--S14 DAG 的 resident。下一任务组开始写 L1/UB 前，必须确认当前任务组中所有
与目标地址重叠的 tensor 都已完成真实末次消费。正式完整实现默认同一核上的任务组
按 S0--S14 完整闭环后再进入下一组；如果阶段性调试只实现到某个中间 Stage，则该
Stage 的调试输出不得占用下一任务组会提前重写的 resident 地址，应放入当前未冲突
的独立地址，或只写 GM 调试输出。不能把“后续 Stage 尚未实现”等同于 tensor
生命周期已经结束。

### Stage 0：Vector，gate/beta 系数与 prepare Cube 输入

```text
g_exp[i]  = exp2(g[i])                    # g_exp: [BT]
g_last    = g_exp[M-1]                    # 数学上为 scalar；实现中广播保存为 [BT]
gate_dA[i,j] = exp2(g[i] - g[j])           # gate_dA: [BT,BT]
decay[i]  = exp2(-g[i] + g[M-1])           # decay: [BT]
bg[i]     = beta[i] * g_exp[i]             # bg: [BT]
kbg       = k * bg[:,None]                 # kbg: [BT,K]
vb        = v * beta[:,None]                # vb: [BT,V]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；S0 搬入，保留至 S14
UB[32,64)      v[2]；BF16；S0 搬入，保留至 S6
UB[64,80)      gate_dA[2]；BF16；S0 生成，保留至 S12
UB[80,112)     vb[2]；BF16；S0 生成，写入 GM 后释放
UB[112,128)    dA_u[2] 后续输出槽；BF16；S1 生成，S2 原位处理后保留至 S4
UB[128,160)    kbg[2]；BF16；S0 生成，写入 GM 后释放
UB[160,245)    空闲；无数据；S0 可复用
UB[245,245.5)  g[2]；FP32；S0 搬入，S0 结束后释放
UB[245.5,246)  beta[2]；FP32；S0 搬入，保留至 S12
UB[246,246.5)  g_exp[2]；FP32；S0 生成，保留至 S12
UB[246.5,247)  g_last[2]；FP32；S0 生成，保留至 S10
UB[247,247.5)  decay[2]；FP32；S0 生成，保留至 S14
UB[247.5,248)  bg[2]；FP32；S0 生成，保留至 S12
```



当前 AIV 最多承包 2 个交错 HV，因此 `g/beta/g_exp/g_last/decay/bg/gate_dA`
各预留 2 份 UB 空间。
当前只设计 Stage 和容量，不讨论精度；`gate_dA` 以及
后续大型跨 Stage 中间量统一按 BF16 驻留。`k/v` 在本 Stage 首次进入 Vector 路径后，分别以
BF16 `[2,BT,K]` 和 `[2,BT,V]` 保留在 UB；`k` 保留到 Stage 14，
`v` 保留到 Stage 6。`beta/g_exp/bg/gate_dA` 保留到 Stage 12，其中
`gate_dA` 和 `bg` 均由 Stage 0 计算一次，后续 Vector Stage 直接复用。
原始 `g` 完成 Stage 0 的全部公式后释放。

操作流程：

1. 首先只搬入当前一个 HV 的完整 `g/beta/k/v`。处理当前
   ping/pong slot 时，MTE2 将本 AIV 的下一个交错 HV 搬入另一
   slot。覆盖下一 slot 前等待该 slot 的 `MTE3_MTE2`；每次搬运只对应一个 HV，不合并两个 head；尾块仍分配完整
   `[BT,...]` 逻辑 buffer，无效行在 VF 中通过有效长度 `M` 屏蔽。
2. 每次 VF 严格只处理一个 HV，完整计算该 HV 的
   `g_exp/g_last/gate_dA/decay/bg/kbg/vb`。当前 VF 结束后
   `streamSlot` 取反，下一次 VF 等待并消费另一份
   `MTE2_V` flag 对应的 slot。
   VF 首先将 `g/beta` 转换为 FP32 `gReg/betaReg` 并生成 `bgReg`；
   后续逐行因子使用 RegTensor Gather 直接从 `gReg/betaReg/bgReg`
   广播，不重复计算公共因子。
   需要跨 Stage 的
   `beta/g_exp/g_last/decay/bg/gate_dA/k/v` 直接写入各自跨 Stage 保存地址；仅供后续
   Cube 使用的 `vb/kbg` 写入本 Stage 临时地址。
3. `kbg[2,BT,K] + vb[2,BT,V]` 按 BF16 共占 64 KiB。S0 自身的大型 tensor 为
   `gate_dA 16 KiB + k 32 KiB + v 32 KiB + kbg/vb 64 KiB = 144 KiB`。
   计入无执行顺序约束的 Stage 1 写入 `dA_u[2]` 后，大型数据占用连续的
   `UB[0,160)`；3 KiB FP32 小向量从 UB 尾端反向排布，中间仅保留一个连续的
   `UB[160,245)` 可复用区。
   每个 HV 的 VF 完成后，将当前 `vb/kbg` 写入 GM。当前 MTE3 读取一份结果时，下一份 VF 可使用
   另一 slot，两份 slot 的 `MTE3_MTE2` 事件独立闭环。
4. `vb/kbg` 完成 GM 搬运后均不在 UB 保留。公共因子
   `bg=beta*g_exp` 保留在 UB 小向量地址，后续 Vector 公式直接复用。

### Stage 1：Cube，DW 与 dA_u

```text
dw0  = du @ h                               # dw0: [BT,K]
dA_u = du @ vb.T                            # dA_u: [BT,BT]
```

两条矩阵乘只共享只读 `du`，彼此不读取本 Stage 输出。

空间布局（L1，单位 KiB）：

```text
L1[0,128)    空闲；无数据；S1 可复用
L1[128,256)  h[4]；BF16；S1 搬入，保留至 S11
L1[256,320)  du[4]；BF16；S1 搬入，保留至 S3
L1[320,384)  kbg[4]；BF16；S3 从 GM 搬入，S3 消费后释放
L1[384,448)  vb[4]；BF16；S1 从 GM 搬入，S1 消费后释放
L1[448,512)  dw0[4]；BF16；S1 生成，保留至 S7
```

操作流程：

1. `vb` 从 Stage 0 的 GM 结果搬入 L1。先搬入当前 HEAD 的
   `du/h`；当前 HEAD 进入 MTE1/Cube 时，MTE2 将下一 HEAD 的
   `du/h` 提前搬入另一 resident，使输入搬运与当前矩阵乘重叠。
2. Cube 完成两条独立矩阵乘。L0A/L0B/L0C 各自使用
   独立 ping/pong，每发射一次 MMAD 就对应取反一次，不用同一
   slot 连续承载两条 GEMM。`dw0[4]` 写入 `L1[448,512)`；`dA_u[2]` 写入
   `UB[112,128)`。
3. `dA_u` 的核间握手只使用 mode=`0x2`，每个 HEAD 一组
   release/ready flag。两个 AIV 各 set release，AIC 聚合 wait 一次后，
   Fixpipe 用 `subBlockId` 只写 owner AIV；AIC set ready 一次，两个
   AIV 各 wait 一次后共同归还 release。
4. `vb` 完成末次 Cube 消费后释放。`h/du/dw0` 保留到 Stage 3；其中 `du`
   继续供 `dvb` 使用，不从 GM 重读。

### Stage 2：Vector，dA_u 预处理并提前计算 kb

```text
dA_u_lower = tril(dA_u, diagonal=-1)       # dA_u_lower: [BT,BT]
kb = k * beta[:,None]                      # kb: [BT,K]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] 并发输出槽；BF16；S3 生成，保留至 S6
UB[112,128)    dA_u[2] -> dA_u_lower[2]；BF16；S1 写入，S2 原位处理后保留至 S4
UB[128,136)    dA_w0[HV0] 后续输出槽；BF16；S3 生成，保留至 S4
UB[136,144)    空闲；原 kbg bank 0 的后半段
UB[144,152)    dA_w0[HV1] 后续输出槽；BF16；S3 生成，保留至 S4
UB[152,160)    空闲；原 kbg bank 1 的后半段
UB[160,192)    kb[2]；BF16；S2 生成，写入 GM 后释放
UB[192,245.5)  空闲；无数据；S2 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 本 Stage 读取 Stage 1 的 `dA_u`，因此必须在 Stage 1 完成后执行。单次 VF 完成
   `dA_u -> dA_u_lower` 和 `kb=k*beta`，不执行任何转置。
2. `kb[2]` 从 `UB[160,192)` 写入 GM；搬运完成后释放该 UB 临时地址，Stage 9 再搬入 L1。
3. Stage 3 从 GM 搬入 Stage 0 生成的 `kbg[4]`，不依赖 Stage 2 输出，
   因此 S2/S3 无执行顺序约束。S2 在 `UB[80,112)` 预留 S3 的 `dvb[2]`；
   `dA_w0[2]` 沿用原 `kbg[2]` 的两个 16 KiB bank 起点，分别使用
   `UB[128,136)` 和 `UB[144,152)`，不把两份 8 KiB 矩阵紧密压到同一 bank。

### Stage 3：Cube，dA_w0 与 dvb

```text
dA_w0 = dw0 @ kbg.T                        # dA_w0: [BT,BT]
dvb    = A @ du                             # dvb: [BT,V]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；S3 从 GM 搬入，原址保留至 S7
L1[32,128)   空闲；无数据；S3 可复用
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  du[4]；BF16；保留至 S3，消费后释放
L1[320,384)  kbg[4]；BF16；S0 写入，S3 消费后释放
L1[384,448)  kb[4]；BF16；S2 写入，保留至 S9
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dw0/kbg/du` 读取前序 L1；Cube 将 `kbg[BT,K]` 作为转置的 B 操作数读取，
   不生成实体转置 tensor。`A[4]` 在 Cube 路径中从 GM 一次搬入
   `L1[0,32)`，并原址保留到 Stage 7。
2. 两条矩阵乘彼此独立。Fixpipe 将两份 `dA_w0` 分别写入原 kbg bank 的
   `UB[128,136)`、`UB[144,152)`，将 `dvb[2]` 写入 `UB[80,112)`。
3. `du/kbg` 完成末次 Cube 消费后释放。

### Stage 4：Vector，dA0

```text
dA0 = dA_u_lower - tril(dA_w0, diagonal=-1) # dA0: [BT,BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2]；BF16；S3 写入，保留至 S6
UB[112,128)    dA_u_lower[2]；BF16；保留至 S4，消费后释放
UB[128,136)    dA_w0[HV0] -> dA0[HV0]；BF16；写入 GM 后释放
UB[136,144)    空闲；原 kbg bank 0 的后半段
UB[144,152)    dA_w0[HV1] -> dA0[HV1]；BF16；写入 GM 后释放
UB[152,245.5)  空闲；无数据；S4 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 读取完整 `dA_w0/dA_u_lower`，生成 `dA0`。
2. `dA0[2]` 在两个原 kbg bank 起点原位覆盖 `dA_w0[2]`，再写入 GM；
   释放 `UB[112,152)` 中本 Stage 使用的有效子区。

### Stage 5：Cube，dA1 与 a2

```text
dA1 = dA0 @ A                              # dA1: [BT,BT]
a2  = k @ k.T                              # a2: [BT,BT]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7
L1[32,96)    空闲；无数据；S5 可复用
L1[96,128)   dA0[4] -> dA1[4]；BF16；S5 从 GM 搬入 dA0，原址 Fixpipe dA1 并保留至 S7
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；S5 搬入，保留至 S13
L1[320,384)  空闲；无数据；S5 可复用
L1[384,448)  kb[4]；BF16；保留至 S9
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dA0` 从 GM 一次搬入 `L1[96,128)`；`A` 读取 `L1[0,32)` 的前序保存数据。
   `k` 从 GM 一次搬入 `L1[256,320)` 并保留到 Stage 13；同一份 `A/k` L1 resident
   分别按 L1A/L1B layout 解释后送入两侧 L0，不创建副本。
2. 两条矩阵乘彼此独立。`dA0` 完成 MTE1 读取后，Fixpipe 将 `dA1[4]`
   原址写入 `L1[96,128)`；将 `a2[2]` 写入 `UB[128,144)`。
3. `A/dA1` 均保留到 Stage 7；Stage 6 不读取 `dA1`。

### Stage 6：Vector，提前完成 dv 与 db_v_partial

```text
dv = dvb * beta[:,None]                    # dv: [BT,V]
db_v_partial = rowSum(dvb * v)             # db_v_partial: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6，消费后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] -> dv[2]；BF16；dvb 保留至 S6，dv 写 GM 后释放
UB[112,128)    空闲；无数据；S6 可复用
UB[128,144)    a2[2]；BF16；S5 写入，保留至 S8
UB[144,245)    空闲；无数据；S6 可复用
UB[245,245.5)  db_v_partial[2]；FP32；S6 生成，保留至 S8
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 完成 `db_v_partial` 和 `dvb -> dv`，不读取或处理 `dA1`。
2. `dv` 写入 GM，`db_v_partial` 保留在 UB 到 Stage 8，释放 `v/dvb` 地址。
   Stage 7 使用独立的 `UB[192,240)` 输出区，与本 Stage 无执行顺序约束。

### Stage 7：Cube，dA2 与 dkbg0

```text
dA2 = A @ dA1                              # dA2: [BT,BT]
dkbg0 = A @ dw0                            # dkbg0: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7，消费后释放
L1[32,96)    空闲；无数据；S7 可复用
L1[96,128)   dA1[4]；BF16；S5 Fixpipe 写入，S7 消费后释放
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  空闲；无数据；S7 可复用
L1[384,448)  kb[4]；BF16；保留至 S9
L1[448,512)  dw0[4]；BF16；保留至 S7，消费后释放
```

操作流程：

1. 两条矩阵乘只共享只读 `A`，彼此不依赖。
2. Fixpipe 将 `dA2[2]` 写入 `UB[192,208)`，将 `dkbg0[2]` 写入
   `UB[208,240)`；两块均不与 Stage 6 的 UB 数据重叠。
3. `dA1/A/dw0` 完成末次 Cube 消费后释放对应 L1 地址。

### Stage 8：Vector，prepare G 完整合成

```text
dA = tril(-(dA2 * gate_dA), diagonal=-1)   # dA: [BT,BT]

AdA = dA * (a2 * beta[:,None])             # AdA: [BT,BT]
dg_A = rowSum(AdA) - colSum(AdA)           # dg_A: [BT]

dg0 = -rowSum(dkbg0 * k * bg[:,None])       # dg0: [BT]
db0 = -rowSum(dkbg0*k*g_exp[:,None])       # db0: [BT]
db_v = db0 + db_v_partial                  # db_v: [BT]
dg_prepare = dg0 + dg_A                    # dg_prepare: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      空闲；无数据；S8 可复用
UB[64,80)      gate_dA[2]；BF16；S0 生成，保留至 S12
UB[80,112)     空闲；无数据；S8 可复用
UB[112,128)    空闲；无数据；S8 可复用
UB[128,144)    a2[2]；BF16；保留至 S8，消费后释放
UB[144,192)    空闲；无数据；S8 可复用
UB[192,208)    dA2[2] -> dA[2]；BF16；dA2 保留至 S8，dA 写入 GM 后释放
UB[208,240)    dkbg0[2]；BF16；S7 写入，保留至 S12
UB[240,244.5)  空闲；无数据；S8 可复用
UB[244.5,245)  db_v_partial[2] -> db_v[2]；FP32；db_v 保留至 S12
UB[245,245.5)  dg_A[2] -> dg_prepare[2]；FP32；dg_prepare 保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 一次 VF 允许读取本 Stage 前序向量公式的结果，因此依次完成上述全部公式。
2. `dA[2]` 写入 GM，供 Stage 9 搬入 L1；`kb[4]` 已由 Stage 2 保留在
   `L1[384,448)`，本 Stage 不重复计算或搬运。
3. `gate_dA` 和 `dkbg0` 保持首次分配的物理地址不变。保留
   `k/gate_dA/dkbg0/bg/db_v/dg_prepare`，释放 `dA/a2` 的 UB 地址。

### Stage 9：Cube，prepare K 矩阵乘

```text
dkb   = dA @ k                             # dkb: [BT,K]
dkb_t = dA.T @ kb                          # dkb_t: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,96)     空闲；无数据；S9 可复用
L1[96,128)   dA[4]；BF16；S9 从 GM 搬入，S9 消费后释放
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  空闲；无数据；S9 可复用
L1[384,448)  kb[4]；BF16；S9 从 GM 搬入，S9 消费后释放
L1[448,512)  空闲；无数据；S9 可复用
```

操作流程：

1. 两条矩阵乘只共享只读 `dA`，彼此不依赖。第二条使用恒等式
   `(kb.T @ dA).T = dA.T @ kb`，只保留左操作数转置语义，不对 Cube 结果转置。
2. 令当前 `(chunk,task-group)` 的临时区基址为 `W9`：`dkb[2]` 写入
   `GM[W9+0,W9+32 KiB)`，`dkb_t[2]` 写入 `GM[W9+32 KiB,W9+64 KiB)`。
3. `dA/kb` 完成末次 Cube 消费后释放 L1 地址。

### Stage 10：Vector，state 归约

```text
state_term = sum(h * dh) * g_last          # state_term: scalar per HV
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      h[0]；BF16；S10 搬入，S10 归约后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     h[1]；BF16；S10 搬入，S10 归约后释放
UB[112,144)    dh[0]；BF16；S10 搬入，S10 归约后释放
UB[144,176)    dh[1]；BF16；S10 搬入，S10 归约后释放
UB[176,208)    空闲；无数据；S10 可复用
UB[208,240)    dkbg0[2]；BF16；保留至 S12
UB[240,244.5)  空闲；无数据；S10 可复用
UB[244.5,245)  db_v[2]；FP32；保留至 S12
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2] -> state_term[2]；FP32；g_last 保留至 S10，state_term 保留至 S14
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. `h/dh` 在 Vector 路径中各从 GM 完整搬入一次。S10 的四份 32 KiB
   slot 不覆盖 S8 的 MTE3 源区，因此不等待 S8 的 `MTE3_MTE2`；复用区只按
   `V_MTE2` 生命周期在 S0、S8、S10 之间闭环。
2. 单次 VF 完成完整归约，`state_term[2]` 覆盖 `g_last` 子槽；释放 `h/dh` UB。

### Stage 11：Cube，DS/DQ 基础矩阵乘

```text
ds0 = do @ v_new.T                         # ds0: [BT,BT]
dq0 = do @ h                               # dq0: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     do 当前 Stage 输入[2]；BF16；S11 从 GM 搬入，S11 结束后释放
L1[32,128)   空闲；无数据；S11 可复用
L1[128,256)  h[4]；BF16；保留至 S11，消费后释放
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  v_new[4]；BF16；S11 搬入，保留至 S13
L1[384,512)  空闲；无数据；S11 可复用
```

操作流程：

1. `h/k` 读取 L1 保存数据；`v_new` 从 GM 一次搬入 `L1[320,384)`；`do` 一次
   搬入 `L1[0,32)`。两条矩阵乘只共享只读 `do`，彼此不依赖。
2. 令 GM 临时区基址为 `W11`：`ds0[2]` 写入 `GM[W11+0,W11+16 KiB)`，
   `dq0[2]` 写入 `GM[W11+16 KiB,W11+48 KiB)`。
3. `h` 完成末次 Cube 消费后释放；`k/v_new` 保留到 Stage 13。

### Stage 12：Vector，prepare K、DS/DQ 后处理

```text
dk_prepare_hv = -dkbg0 * bg[:,None]
                + dkb * beta[:,None] + dkb_t           # dk_prepare_hv: [BT,K]
dk_prepare[hk] = sum(dk_prepare_hv[hv_i], hv_i in H(hk)) # dk_prepare: [BT,K]
db_prepare = db_v + rowSum(dkb * k)                    # db_prepare: [BT]

dq_base = dq0 * g_exp[:,None] * scale                  # dq_base: [BT,K]
ds = tril(ds0 * gate_dA) * scale                       # ds: [BT,BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      空闲；无数据；S12 可复用
UB[64,80)      gate_dA[2]；BF16；S0 生成，S12 消费后释放
UB[32,64)      dkb[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[80,112)     dkb_t[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[112,128)    ds0[2] -> ds[2]；BF16；S12 从 GM 搬入并原位生成，写入 GM 后释放
UB[128,176)    空闲；无数据；S12 可复用
UB[176,208)    dq0[2] -> dq_base[2]；BF16；S12 从 GM 搬入并原位生成，dq_base 保留至 S14
UB[208,240)    dkbg0[2] -> dk_prepare[2]；BF16；S12 原位生成，dk_prepare 保留至 S14
UB[240,244)    空闲；无数据；S12 可复用
UB[244,244.5)  beta_raw[2] -> dbeta[2]；FP32；S12 搬入，写 GM 后释放
UB[244.5,245)  db_v[2] -> db_prepare[2]；FP32；S12 生成 dbeta 后释放
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12，消费后释放
UB[246,246.5)  g_exp[2]；FP32；保留至 S12，消费后释放
UB[246.5,247)  state_term[2]；FP32；保留至 S14
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；S12 消费后释放
```

操作流程：

1. `dkb/dkb_t/ds0/dq0/beta_raw` 从上述 GM 临时区各完整搬入一次，所有输入在
   一次 VF 前到齐。
2. 一次 VF 完成两组向量公式；`dk_prepare` 原位覆盖 `dkbg0`，`dq_base` 原位覆盖
   `dq0`。`db_prepare` 完成 beta backward 后将 `dbeta` 写入 GM。
3. `ds[2]` 原位覆盖 `ds0[2]`，再写入 GM，供 Stage 13 搬入 L1。Stage 12 是
   `gate_dA/bg` 的最后一个消费者，完成后释放二者；保留
   `k/dk_prepare/dq_base/dg_prepare/state_term/decay`；这些数据均保持原物理地址。

### Stage 13：Cube，DQ/DK 最终矩阵乘

```text
dq_intra = ds @ k                          # dq_intra: [BT,K]
dk_intra = ds.T @ q                        # dk_intra: [BT,K]
dk0      = v_new @ dh                      # dk0: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     q 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[32,96)    dh 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[96,128)   ds[4]；BF16；S13 从 GM 搬入，S13 消费后释放
L1[128,256)  空闲；无数据；S13 可复用
L1[256,320)  k[4]；BF16；保留至 S13，消费后释放
L1[320,384)  v_new[4]；BF16；保留至 S13，消费后释放
L1[384,512)  空闲；无数据；S13 可复用
```

操作流程：

1. Stage 13 整体依赖 Stage 12 的 `ds`。`ds/q/dh` 在 Cube 路径中各从 GM 完整
   搬入一次。
2. 三条矩阵乘彼此独立。Fixpipe 将 `dk0[2]`、`dq_intra[2]`、`dk_intra[2]`
   分别写入 `UB[32,64)`、`UB[112,144)`、`UB[144,176)`。
3. 释放全部 L1 数据。

### Stage 14：Vector，DQ/DK 与 gate 最终 backward

```text
dq_hv = dq_base + dq_intra                 # dq_hv: [BT,K]
dg_chunk_partial = rowSum(dq_hv * q)       # dg_chunk_partial: [BT]
dg_chunk_partial[M-1] += state_term        # dg_chunk_partial: [BT]
dq_chunk[hk] = sum(dq_hv[hv_i], hv_i in H(hk)) # dq_chunk: [BT,K]
dq = dq_chunk * q_rstd[:,None]
     - rowSum(dq_chunk*q)[:,None] * q * q_rstd[:,None] # dq: [BT,K]

dk_base = dk0 * decay[:,None]              # dk_base: [BT,K]
dk_hv = dk_base + dk_intra                 # dk_hv: [BT,K]
dk_base_dot = rowSum(dk_base * k)          # dk_base_dot: [BT]
dg_chunk = dg_chunk_partial - dk_base_dot
           - rowSum(dk_intra * k)          # dg_chunk: [BT]
dg_chunk[M-1] += sum(dk_base_dot)          # dg_chunk: [BT]
dk_chunk[hk] = sum(dk_hv[hv_i], hv_i in H(hk)) # dk_chunk: [BT,K]
dk_raw = dk_chunk + dk_prepare             # dk_raw: [BT,K]
dk = dk_raw * k_rstd[:,None]
     - rowSum(dk_raw*k)[:,None] * k * k_rstd[:,None] # dk: [BT,K]

x[t] = dg_chunk[t] + dg_prepare[t]         # x: [BT]
dg_raw[t] = sum(x[j], j=t..M-1)            # dg_raw: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14，计算结束后释放
UB[32,64)      dk0[2] -> dk_base[2] -> dk_hv[2] -> dk[2]；BF16；S13 写入，S14 写 GM 后释放
UB[64,80)      空闲；无数据；S14 可复用
UB[80,112)     q[2]；BF16；S14 从 GM 搬入，S14 后释放
UB[112,144)    dq_intra[2]；BF16；S13 写入，S14 消费后释放
UB[144,176)    dk_intra[2]；BF16；S13 写入，S14 消费后释放
UB[176,208)    dq_base[2] -> dq[2]；BF16；保留至 S14，写 GM 后释放
UB[208,240)    dk_prepare[2]；BF16；保留至 S14，消费后释放
UB[240,244)    空闲；无数据；S14 可复用
UB[244,244.5)  q_rstd[2] -> dk_base_dot[2]；FP32；S14 搬入并原位复用，S14 后释放
UB[244.5,245)  k_rstd[2]；FP32；S14 搬入，S14 后释放
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14，消费后释放
UB[245.5,246)  g_input[2]；FP32；S14 搬入，S14 后释放
UB[246,246.5)  dg_chunk_partial[2]；FP32；S14 生成，S14 后释放
UB[246.5,247)  state_term[2]；FP32；保留至 S14，消费后释放
UB[247,247.5)  decay[2]；FP32；保留至 S14，消费后释放
UB[247.5,248)  dg_raw[2]；FP32；S14 生成，gate backward 后释放
```

操作流程：

1. `q/q_rstd/k_rstd/g_input` 在 Vector 路径中各从 GM 完整搬入一次。
2. 单次 VF 依次完成全部 DQ、DK、suffix sum 和 gate backward 公式。DQ 归一化
   完成后，`q_rstd` 子槽原位复用为 `dk_base_dot`；尾元素补偿直接归约该驻留结果，
   不重复计算 `dk_base*k`。
3. 最终 `dq/dk/dg/dA_log/ddt_bias` 写入 GM，释放全部 UB 数据。

关闭 `use_qk_l2norm_in_kernel` 时：

```text
dq = dq_chunk                                 # dq: [BT,K]
dk = dk_raw                                   # dk: [BT,K]
```

beta sigmoid backward：

```text
s     = sigmoid(beta_raw)                     # s: [BT]
dbeta = db_prepare * s * (1-s)                # dbeta: [BT]
```

关闭 `use_beta_sigmoid_in_kernel` 时：

```text
dbeta = db_prepare                            # dbeta: [BT]
```

GDN gate backward：

```text
z          = g_input + dt_bias[hv]             # z: [BT]，dt_bias 可选
neg_exp_A  = -exp(A_log[hv])                   # neg_exp_A: scalar
gate_y     = neg_exp_A * softplus(z)           # gate_y: [BT]
dg         = neg_exp_A * sigmoid(z) * dg_raw   # dg: [BT]

dA_partial[chunk,hv] = sum(dg_raw * gate_y)    # scalar
dt_partial[chunk,hv] = sum(dg)                 # scalar
```

全局参数梯度：

```text
dA_log[hv]   = sum_chunk(dA_partial[:,hv])     # scalar per HV
ddt_bias[hv] = sum_chunk(dt_partial[:,hv])     # scalar per HV
```

关闭 `use_gate_in_kernel` 时：

```text
dg = dg_raw                                   # dg: [BT]
dA_log = None
ddt_bias = None
```

## 5. 新 Stage 资源分配方案

本节只规定数据驻留和 stage 类型，不规定具体 EventID、workspace offset 或 VF 内部
指令排布。每个 stage 只能执行 Cube 矩阵乘或 Vector 向量操作中的一种。L1 只作为
Cube 的输入/中间结果空间，UB 只作为 Vector 运算空间以及 Cube-to-Vector 结果落点。
Vector 不在 L1 上计算，Cube 不从 UB 取矩阵乘输入。
GM/L1/UB 之间的数据搬运和必要同步不改变 Stage 的计算类型：Cube Stage 只能发出
矩阵乘，Vector Stage 只能发出向量指令。

本方案必须同时满足以下固定约束：

1. 每个 Stage 只能包含 Cube 矩阵乘或 Vector 向量操作之一。
2. 同一 Cube Stage 内的矩阵乘不能读取该 Stage 内任一 Cube/Vector 操作刚生成的
   输出，即同一 Cube Stage 内的矩阵乘必须彼此独立。Vector Stage 允许后续向量公式
   读取该 Stage 内前序向量公式的输出。一个 Stage 完成后，其输出才视为可供后续
   Stage 依赖的数据。
3. L1 固定为 512 KiB 且只供 Cube 使用；UB 固定为 248 KiB 且只供 Vector 使用。
4. Cube-to-Vector 非算子输出必须写入两份 UB 保存空间。
5. Vector-to-Vector 非算子输出必须保留两份 UB 保存空间。
6. Vector-to-Cube 非算子输出必须写入四份 L1 保存空间。
7. Cube-to-Cube 非算子输出必须保留四份 L1 保存空间。
8. UB 不预留固定子区域，完整 `UB[0,248)` 均可由当前 Vector Stage 使用；跨 Stage
   保存数据、当前 Stage 输入、临时量和输出的同时存活总量不得超过 248 KiB。
9. L1 不预留固定子区域，完整 `L1[0,512)` 均可由当前 Cube Stage 使用，并可在不同
   Stage 改变地址语义；需要跨 Stage 保存的 tensor 必须按四份容量预留。
10. 矩阵乘路径和向量路径内部均不得从 GM 重复搬运同一数据。若同一原始输入同时参与
    Cube 和 Vector，允许分别执行一次 `GM -> L1` 和一次 `GM -> UB`。
11. UB 内任一地址在前一数据完成真实末次消费后都可改变语义，不存在固定
    tensor、输入、临时量或小向量固定边界。
12. Vector Stage 不允许拆分 VF pass 或 tile。进入 VF 前必须一次搬完本 Stage 的完整
    逻辑输入，随后通过一次 VF 调用完成该 Stage 的全部向量公式；尾块仍分配完整
    `BT=64` buffer，通过 `M` 屏蔽无效元素。
13. 没有依赖关系的 Cube Stage 和 Vector Stage 不建立执行顺序约束，允许并发执行；
    因此两者同时存活的数据地址不能重叠。地址只有在前一 tensor 的真实末次
    消费者完成后才能复用，不能依据 Stage 编号先后复用无依赖数据的地址。
14. 若第 1--13 条无法同时满足，允许通过 GM workspace 落地并再次搬入解决容量冲突。
    回退只用于经过完整依赖 DAG 和容量计算证明无法片上驻留的 tensor，并且每次写入、
    读取的对象和 GM 相对偏移都必须在对应 Stage 中明确列出。
15. 优先避免连续 Cube Stage 或连续 Vector Stage。只要能前移或后移真实计算任务
    作为中间 Stage，就允许增加 Stage 数；不得插入无公式、无输出的空 Stage。若连续
    同类型 Stage 无法消除，必须均衡可调度任务，使连续段内每个 Stage 的任务尽可能少。
16. Vector 路径尽量避免重复计算相同中间量；只要容量允许，重复使用的向量结果应按
    两份容量保留在 UB，直到最后一个 Vector 消费者完成。当前方案将
    `gate_dA` 和 `bg=beta*g_exp` 保留到 Stage 12，不重新计算；Stage 14 将
    `dk_base_dot=rowSum(dk_base*k)` 保存在 UB 小向量地址，供 `dg_chunk` 与尾元素
    补偿共同读取。
17. 减少 Stage 总数的优先级最低。只有在第 1--16 条均已满足、不会重新形成连续
    同类型 Stage、也不会增加连续同类型 Stage 的任务数时，才允许合并 Stage。
18. 任一 tensor 一旦在 UB 或 L1 分配物理区间，从首次写入到真实末次消费完成，
    其物理区间必须保持不变，禁止中途搬移、压紧或重排。只允许在原地址原位更新
    tensor 语义，或在原 tensor 生命周期彻底结束后将该地址分配给新 tensor。

完整 DAG 核算中，S10 的全部 UB 活跃数据已占 211.5 KiB。无依赖的 S9 若将
`dkb/dkb_t` 直接写 UB，需要额外 64 KiB；S11 若将 `ds0/dq0` 直接写 UB，需要额外
48 KiB，两种并发集合分别为 275.5 KiB 和 259.5 KiB，均超过完整 248 KiB UB。因此两组结果
使用第 14 条容量回退：S9 写 `W9`，S11 写 `W11`，S12 再各完整读取一次。
S10 与 S9/S11 不增加执行顺序约束。

### 5.1 Stage 序列

```text
S0  Vector : gate/beta 系数、kbg、vb
S1  Cube   : dw0、dAu
S2  Vector : dAuLower、kb
S3  Cube   : dAw0、dvb
S4  Vector : dA0
S5  Cube   : dA1、a2
S6  Vector : dv、dbVPartial
S7  Cube   : dA2、dkbg0
S8  Vector : dA、dbV、dgPrepare
S9  Cube   : dkb、dkbT
S10 Vector : stateTerm
S11 Cube   : ds0、dq0
S12 Vector : dkPrepare、dbPrepare、ds、dqBase
S13 Cube   : dqIntra、dkIntra、dk0
S14 Vector : dq、dk、dgChunk、dgRaw 和 gate backward
```

Cube Stage 输入依赖检查：

```text
S1  dw0  <- du,h          dAu <- du,S0.vb
S3  dAw0 <- S1.dw0,S0.kbg  dvb <- A,S1.du
S5  dA1  <- S4.dA0,A      a2  <- k,k
S7  dA2 <- A,S5.dA1       dkbg0 <- A,S1.dw0
S9  dkb  <- S8.dA,k       dkbT <- S2.kb,S8.dA
S11 ds0  <- do,v_new      dq0 <- do,h
S13 dqIntra <- S12.ds,k   dkIntra <- S12.ds,q   dk0 <- v_new,dh
```

同一行中的多个矩阵乘只共享只读输入，互不读取本 Stage 产生的结果；存在结果依赖的
`dw0 -> dAw0`、`dA0 -> dA1 -> dA2` 已拆到不同 Stage。当前 15 个 Stage 从 S0
到 S14 严格按 Vector/Cube 交替，没有连续 Cube 或连续 Vector。S2 前移 `kb`，并用
`dA_u_lower` 建立对 S1 的依赖；S3 直接读取 S0 保存在 L1 的 `kbg`，因此 S2/S3
无执行顺序约束，对应 Cube 输出槽已计入 S2 的 UB 并发集合。S5 将 `dA1`
直接保留在 L1 供 S7 使用；S6 完成 `dv/db_v_partial`，其 UB 地址与 S7 输出不重叠，
因此 S6 和 S7 没有执行顺序约束。负号延后到 S8 与 `gate_dA` 的乘法融合。
两个插入的 Vector 都承担真实公式，不是空 Stage。Stage 数仅在以上条件满足后再缩减。

Cube 输出只要后续由 Vector 消费且不是算子输出，容量可行时由 Fixpipe 直接写入两份
UB 常驻槽位；容量不可行时允许按第 14 条规则经 GM workspace 中转一次。Vector 输出
只要后续由 Cube 消费且不是算子输出，就从
UB 直接搬入 4 份 L1 保存空间，不经过 GM 中转。Cube 输出只要后续仍由 Cube 消费且
不是算子输出，就直接写入 4 份 L1 保存空间。Vector 输出只要后续仍由 Vector 消费且
不是算子输出，就保留在两份 UB 保存空间。当前方案中的 `dw0/dA1/kb` 走 L1
保存空间，`gate_dA/bg/k/v/state_term/dk_prepare/dq_base/dg_prepare` 走 UB 保存空间。

### 5.2 L1 全空间分配

```text
L1_total = 512 KiB
L1_fixed_reservation = 0 KiB
L1_available_per_cube_stage = 512 KiB
```

完整 `L1[0,512)` 均可存放跨 Stage 保存数据或当前 Cube Stage 输入。地址可随 Stage
改变语义；只有仍有后续 Cube 消费者的数据不可覆盖。凡是跨 Stage 保存的 tensor，
必须按最多 4 个 HV 预留 4 份容量。按 BF16 Cube 输入计算，各关键 Stage 的完整 L1
同时存活量为：

```text
S0 end  : vb[4] + kbg[4]                                = 128 KiB
S1 peak : h[4] + du[4] + vb[4] + kbg[4] + dw0[4]       = 384 KiB
S1 end  : h[4] + du[4] + kbg[4] + dw0[4]               = 320 KiB
S2 end  : h[4] + du[4] + kbg[4] + kb[4] + dw0[4]       = 384 KiB
S3 peak : h[4] + du[4] + kbg[4] + kb[4] + dw0[4]
          + A[4]                                       = 416 KiB
S3 end  : h[4] + A[4] + kb[4] + dw0[4]             = 288 KiB
S4 end  : h[4] + A[4] + kb[4] + dw0[4]              = 288 KiB
S5/S6   : h[4] + k[4] + A[4] + (dA0 -> dA1)[4] + dw0[4]
                                                        = 320 KiB
S7 end  : h[4] + k[4] + kb[4]                       = 256 KiB
S8/S9   : h[4] + k[4] + dA临时输入[4] + kb[4]      = 288 KiB
S9 end  : h[4] + k[4]                               = 192 KiB
S11 peak : h[4] + k[4] + v_new[4] + do 当前输入[2] = 288 KiB
S11 end : k[4] + v_new[4]                           = 128 KiB
S13 peak : k[4] + v_new[4] + ds临时输入[4] + q[2] + dh[2] = 256 KiB
```

L1 完整峰值为 S3 的 416 KiB，剩余 96 KiB。不存在固定不可借用区域。

S3 中 `A[4]` 一次搬入 `L1[0,32)` 并原址保留到 S7。`k[4]` 在 S5 首次搬入
`L1[256,320)` 并保留到 S13。`kb[4]` 从 S2 保留到 S9，`h` 从 S1 保留到 S11，
`A/dw0` 保留到 S7，`v_new` 从 S11 保留到 S13。
`L1[96,128)` 作为 Cube Stage 从 GM 搬入短生命周期输入的临时区，严格按
`dA0(S5) -> dA1(S5--S7) -> dA(S9) -> ds(S13)`
的顺序复用。`dA1` 由 S5 原址 Fixpipe 生成，其余 tensor 在对应 Cube Stage 内搬入。该地址
不与下一任务组 S0 的 `kbg=L1[320,384)` 或 S1 的 `vb=L1[384,448)` 重叠。

任何原始输入在对应计算路径中
均只从 GM 读取一次。

#### 5.2.1 L1 生命周期与任务组边界检查

逐 Stage 检查后的关键 L1 生命周期如下：

| 地址 | tensor 生命周期 | 后续复用条件 |
|---|---|---|
| `L1[0,32)` | `A`：S3 写，S5/S7 读 | S7 完成后才可复用 |
| `L1[96,128)` | `dA0` 由 S5 从 GM 搬入并原址覆盖为 `dA1`，之后与 `dA`、`ds` 顺序复用 | 每个 tensor 的消费 Stage 完成后才可覆盖 |
| `L1[128,256)` | `h`：S1 写，S11 读 | S11 完成后才可复用 |
| `L1[256,320)` | `k`：S5 写，S13 读 | S13 完成后才可复用 |
| `L1[320,384)` | `kbg`：S0--S3；`v_new`：S11--S13 | 前一语义末次消费后才可切换 |
| `L1[384,448)` | `vb`：S0--S1；`kb`：S2--S9 | 前一语义末次消费后才可切换 |
| `L1[448,512)` | `dw0`：S1--S7 | S7 完成后才可复用 |

检查结论：同一任务组内部不存在仍未消费就被后续 Stage 覆盖的 L1 地址。短生命周期
Cube 输入从 GM 搬入 `L1[96,128)`，不会与下一任务组 S0 `kbg` 地址重叠。但长生命周期 resident 仍有意复用下一任务组的 S0/S1 地址，
例如 `v_new`、`kb` 和 `dw0`。因此当前单份 resident 设计只允许同一任务组完成
S0--S14 完整闭环后再启动下一任务组；若未来需要任务组间重叠执行，必须为所有跨组
冲突的 resident 增加独立 bank 或改用 GM workspace，不能只调整核间 flag。

### 5.3 UB 全空间分配

完整 UB 容量：

```text
UB_total = 248 KiB
UB_fixed_reservation = 0 KiB
UB_available_per_vector_stage = 248 KiB
```

当前布局中，大型 BF16 tensor 从 UB 低地址向上紧凑排列，FP32 小向量从
`UB[248)` 向下紧凑排列。`bg` 从 S0 保留到 S12，使用 `UB[247.5,248)`；
`gLast` 在 S10 完成末次消费后由 `stateTerm[2]` 原位覆盖。两端排布只决定首次
分配地址；任何跨 Stage 数据在生命周期内均保持该地址不变。地址只有在原数据完成
真实末次消费后才可分配给新数据。

每个跨 Stage Vector 中间量以及 Cube-to-Vector 结果，都按两个 HV 分配两份对应
shape 的 UB 空间。大型 tensor 按 BF16 保存，小向量和归约标量按 FP32 保存。
按依赖 DAG 计算的完整 UB 活跃集合为：

```text
S0 VF peak:
    S0 Vector 大型 tensor 144 KiB + FP32 小向量 3 KiB    = 147 KiB
S2 与 S3 并发峰值:
    gateDA[2] + k[2] + v[2] + dAuLower[2] + kb[2]
    + S3.dAw0[2] + S3.dvb[2]                             = 176 KiB
    FP32 小向量                                           = 2.5 KiB
    S2/S3 total                                           = 178.5 KiB
S3 -> S4:
    gateDA[2] + k[2] + v[2] + dAw0[2] + dAuLower[2]
    + dvb[2]                                              = 144 KiB
    FP32 小向量                                           = 2.5 KiB
    S4 total                                              = 146.5 KiB
S5 -> S6:
    gateDA[2] + k[2] + v[2] + dvb[2] + a2[2]             = 128 KiB
    FP32 小向量（含 dbVPartial）                           = 3 KiB
    S6 total                                              = 147 KiB
S7 -> S8:
    gateDA[2] + k[2] + dA2[2] + a2[2] + dkbg0[2]         = 112 KiB
    FP32 小向量                                           = 3.5 KiB
    S8 total                                              = 115.5 KiB
S10 与 S9/S11 并发（两组 Cube 输出写 GM workspace）:
    大型 tensor 208 KiB + FP32 小向量 3.5 KiB           = 211.5 KiB
S12:
    k[2] + gateDA[2] + (dkbg0[2] -> dkPrepare[2])
    + dkb[2] + dkbT[2] + (ds0[2] -> ds[2])
    + (dq0[2] -> dqBase[2])                              = 192 KiB
    FP32 小向量和 betaRaw[2]                              = 4 KiB
    S12 total                                             = 196 KiB
S14:
    k[2] + dkPrepare[2] + dqBase[2] + dqIntra[2]
    + dkIntra[2] + dk0[2] + q[2]                         = 224 KiB
    FP32 小向量                                           = 4 KiB
    S14 total                                             = 228 KiB
```

UB 完整峰值为 S14 的 228 KiB，剩余 20 KiB。不存在固定不可借用区域：

```text
UB_peak = 228 KiB
UB_free_at_peak = 248 KiB - 228 KiB = 20 KiB
```

UB 生命周期检查同样以完整任务组为边界。`gate_dA/k/v/bg` 等从前序 Stage 保留到
后续 Stage 的数据，在当前任务组完成真实末次消费前，不允许被下一任务组 S0 的
输入搬运覆盖。各 Stage 表内的原位覆盖只发生在源 tensor 已完成末次消费之后；
同一任务组内部未发现提前覆盖。当前 UB 也只有一份跨 Stage resident，因此任务组间
重叠执行的限制与 L1 相同：必须先扩充 resident bank 或将跨组数据落到 GM，不能
依赖双缓冲 EventID 或核间 flag 自动解决地址冲突。

每个 Vector Stage 在 VF 调用前一次性完成完整逻辑输入搬运，并通过一次 VF 完成全部
公式。当前 Stage 输入、临时量、跨 Stage 数据和输出共同计入 248 KiB 总量。

各 Stage 的 UB 绝对偏移已经写在对应“空间布局”中。`dkbg0` 由 S7 写入
`UB[208,240)` 并原址保留到 S12；S9 的 `dkb/dkb_t` 和 S11 的 `ds0/dq0` 在 GM workspace
驻留，只在 S12 调用 VF 前搬入 UB。

Vector Stage 的完整 shape 结果按生命周期复用已消费的 UB 地址，例如
`dq0 -> dq_base`、`dkbg0 -> dk_prepare`、
`dk0 -> dk_base -> dk_hv`。
`AdA/dq_hv` 等只在单次 VF 表达式内部使用，不分配跨 Stage 完整 tensor；
S14 将重复使用的 `dkBaseDot[2]` 保存在已释放的 `qRstd` 地址中；行列归约结果使用
当前 Stage 布局中列出的 FP32 地址。

### 5.4 跨 stage 数据保留原则

```text
Cube -> Vector 且容量可行：Fixpipe -> UB[2] 保存空间 -> Vector 消费
Cube -> Vector 且容量不可行：Fixpipe -> GM workspace -> UB[2] -> Vector 消费
Cube -> Cube 且非算子输出：Fixpipe -> L1[4] 保存空间，保留到末次 Cube 消费
Vector -> Cube：UB -> GM -> L1[4]，由消费它的 Cube Stage 完整搬入
Vector -> Vector 且非算子输出：保留 UB[2]，直到末次 Vector 消费
算子最终输出：Vector 从 UB 写 GM
```

矩阵乘输入第一次从 GM 搬入 L1 后，只要后续 Cube 仍会使用，就必须保留在四份 L1
保存空间中；不允许因 stage 切换重复从 GM 搬运。Cube 结果若将由 Vector 使用，
优先落入两份 UB；仅当完整依赖 DAG 的并发活跃集合超过 UB 总容量时，
允许写一次 GM workspace，再由 Vector 完整读取一次。当前方案中的
`dkb/dkb_t`、`ds0/dq0` 使用该回退路径。

Vector 输入第一次从 GM 搬入 UB 后，只要后续 Vector Stage 仍会使用，就必须保留在
两份 UB；不允许因 stage 切换再次从 GM 搬运。当前方案中 `k` 在 S0--S14、
`v` 在 S0--S6、`beta` 在 S0--S12 的 Vector 生命周期内连续驻留；原始 `g` 在 S0
完成全部派生量后释放。
同一原始输入若同时参与 Cube 和 Vector，两条计算路径相互独立：允许各有一次
`GM -> L1` 和一次 `GM -> UB`，但任一路径内部均不得发生第二次 GM 读取。

Stage 切换时，完整 `UB[0,248)` 和 `L1[0,512)` 内已结束生命周期的地址均可改变
tensor 语义。任何复用都必须以真实末次消费者完成为前提。
