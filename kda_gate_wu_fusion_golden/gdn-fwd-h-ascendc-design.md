# GDN/KDA FwdH A5 Ascend C 逐 Stage 算子设计

## 1. 目标

本文给出 `ChunkGatedDeltaRuleFwdH` 在 A5 上的目标实现方案。本文只复用
`gdn-backward-finalize-ascendc-design(1).md` 的章节和空间图表达方式，不继承其中的 backward
公式、Stage 编号或固定内存划分。

设计目标如下：

1. 每个 Stage 只包含 Cube 矩阵运算或 Vector 运算中的一种。
2. Cube Stage 的任一 MMAD 只读取进入该 Stage 前已经存在的数据，不读取本 Stage 的 Cube/Vector 输出。
3. Vector Stage 允许在同一次 VF 内使用本 VF 前面生成的寄存器结果。
4. L1 仅存放 Cube 输入和 Cube 跨 Stage 驻留数据；UB 仅存放 Vector 输入、输出和 Vector 跨 Stage 驻留数据。
5. Cube 到 Vector 的非算子输出以两份 UB slot 保留；Vector 到 Cube 的非算子输出以四份 L1 slot 保留。
6. 一个 Stage 实例完整搬入一个 head 的全部数据，只调用一次完整 VF 或执行一次逻辑 MMAD，不按行分 pass、分 tile 或多次 VF。
7. 不重复从 GM 搬入相同数据；只有容量无法闭合时才按规则 14 使用 GM fallback。
8. UB/L1 采用固定地址分区；已有活跃/驻留 tensor 不得为整理碎片而搬移。派生输出只写入
   设计时确定且运行期不变的独立目标区，或已经完成最后读取的固定子区。
9. 同一模板实例中，同一语义的所有 active head 使用完全相同的计算和存储路径。
10. 主链保持 `Cube -> Vector -> Cube -> Vector`，在不牺牲其他规则的前提下使用最少 Stage。

本文描述的是目标架构，不表示当前 kernel 已经完成这些改造。Stage 实例的原子粒度定义为：

```text
一个 (sequence, chunk, value_head)
```

每个 AIC round 最多调度 4 个 value head；两个 AIV 各负责最多 2 个 head。L1 的 `[4]` 和
UB 的 `[2]` 表示 round 级物理 slot 数，不表示一个 VF 必须同时计算两个 head。每个 head
调用一次完整 VF，VF 覆盖该 head 的整个逻辑 Tensor。

UB 图中的 `head 0/1` 与 `local slot 0/1` 都表示单个 AIV 的两个本地任务槽，不是 round
head 编号；新增布局统一优先使用 `local slot`：

```text
AIV0: local slot 0 -> round head 0；local slot 1 -> round head 2
AIV1: local slot 0 -> round head 1；local slot 1 -> round head 3
AIC : L1 resident slot 0/1/2/3 -> round head 0/1/2/3
```

## 2. 范围与 shape

### 2.1 目标平台与固定规格

```text
SoC                   = A5 / arch35
kernel mode           = 1 AIC : 2 AIV MIX
chunk_size / BT       = 64
K                     = 128
V                     = 128
k、kg、w、u、h、V_new、V_new_g dtype = BF16，2 Byte
GateT                 = BF16 或 FP32
StateT                = BF16 或 FP32
PType                 = BF16（StateT=BF16）或 FP32（StateT=FP32）
VNewType              = BF16
DType                 = FP32
Vector CalcT          = FP32
Cube tl.dot operand   = BF16 x BF16
Cube accumulate       = FP32
```

本设计的容量计算依赖 `BT=64, K=128, V=128`。进入本 A5 方案前，Host 必须完成这些
shape、dtype、gate 模式和属性拦截；kernel 内不增加 `static_assert` 作为用户入参校验。
varlen 模式要求每个 sequence 非空，即所有相邻 `cu_seqlens` 严格递增；Host 对
`cu_seqlens[n+1] <= cu_seqlens[n]` 返回参数错误，kernel 内不存在 `M=0` 的 Stage task。

`k/w/u` 在本 A5 方案中只支持 BF16。gate 独立支持 BF16/FP32；initial/final state 独立
支持 BF16/FP32。Host 必须在进入 kernel 前拒绝 FP16 的 `k/w/u`，下文不保留 FP16 分支。

本文判断矩阵乘精度时，只看进入 `tl.dot`/MMAD 的两个 operand dtype 和 accumulator dtype，
不以 GM/UB 中某个 Tensor 的存储 dtype 代替运算 dtype。两条状态分支的 S0/S2 均使用
`BF16 x BF16 -> FP32 accumulate`：即使 `R_c` 以 FP32 保存，S0 读取的也是
`H_c=cast_BF16(R_c)`，因此该次矩阵乘仍是 BF16 精度边界，不是 FP32 矩阵乘。

S0 与 S2 的跨 Stage 输出使用不同精度合同，不能再用一个 `InterT` 同时描述：

```text
S0: Pacc FP32 -> PType
    StateT=BF16 时 PType=BF16；StateT=FP32 时 PType=FP32

S1: 所有 Vector 算术在 FP32 寄存器完成；进入 S2 前，`V_new` 和 g-only 的 `V_new_g` 固定量化为 BF16

S2: g-only 为 BF16 `k` x BF16 `V_new_g`，gk-only 为 BF16 `kg` x BF16 `V_new`；均为 FP32 D，DType 固定为 FP32，不在 S2 末尾量化

S3: BF16 state分支在 FP32 寄存器完成 gate/add 后才把 Rnext 量化为 BF16；
    FP32 state分支保持 Rnext FP32
```

FP32 state 分支对齐 fla-org 通用 `chunk_delta_h.py` 的运算边界：rolling state 为 FP32，
但 Stage0 在 `tl.dot` 前执行 `b_h.to(b_w.dtype)`，因此 W/H 两个 operand 都是 BF16；
`P` 保持 FP32 参与 `u-P`；Stage2 前把 `b_v` 转成 `k.dtype`，因此 g-only 的 `k`/`V_new_g`
和 gk-only 的 `kg`/`V_new` 两个 operand 也都是 BF16；Stage2 的 FP32 累加结果直接加到
FP32 rolling state。这里明确选择通用实现作为
精度参考，不使用把 W/K 显式转为 FP32 的 `triton_ascend` 专用实现作为本设计合同。对应源码
见 [state加载及Stage0 dot](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py#L119-L200)
和 [Stage2前量化及dot](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py#L206-L278)。

BF16 state 分支只对齐 FlashKDA 的状态递推精度边界：R、P、`V_new`和g-only的`V_new_g`为BF16，矩阵乘输入为BF16、
累加为 FP32，Stage2 的 FP32 D 直接参与 FP32 state update，最后才把 Rnext 量化为 BF16。
对应边界见 FlashKDA 的
[BF16 state accumulator与MMA](https://github.com/MoonshotAI/FlashKDA/blob/master/csrc/smxx/fwd_kernel2.cuh#L67-L106)
和 [FP32更新后回写BF16 state](https://github.com/MoonshotAI/FlashKDA/blob/master/csrc/smxx/fwd_kernel2.cuh#L659-L724)。
该表述不承诺与融合了 `v/beta/INV` 且 chunk 规格不同的 FlashKDA 整算子逐元素或 bitwise 一致。

逐 Stage 精度与保留边界如下。表中的 “dot dtype” 专指进入矩阵乘指令的 operand dtype：

| 数据/运算 | `StateT=BF16`（FlashKDA边界） | `StateT=FP32`（fla-org通用边界） | 最后消费者/保留期 |
| --- | --- | --- | --- |
| `R_c` | BF16 | FP32 | 当前 S3；生成 `R_{c+1}` 后按分支跨 chunk 保留 |
| `H_c` | BF16，与 `R_c` 同值 | BF16，`cast_BF16(R_c)` | L1保留到当前 S0 MTE1 |
| S0 dot | BF16 W x BF16 H，FP32累加 | BF16 W x BF16 H，FP32累加 | Pacc在S0末转换为PType |
| `P_c` | BF16 | FP32 | UB保留到当前 S1 VF 最后一次读取 |
| `V_new_fp32` | FP32寄存器 | FP32寄存器 | 当前 S1 VF；转换为 `V_new` 和 g-only 的 `V_new_g` |
| `V_new` | BF16 | BF16 | 公开 `v_new`；gk-only 的 S2 右操作数 |
| `V_new_g` | BF16 | BF16 | g-only 的 S2 右操作数；由 `V_new` 乘 chunk 内门控因子得到；L1保留到当前 S2 MTE1 |
| S2 dot | g-only 为 BF16 k x BF16 `V_new_g`；gk-only 为 BF16 kg x BF16 `V_new`，均 FP32 累加 | 同左 | Dacc直接成为FP32 D |
| `D_c` | FP32 | FP32 | UB保留到当前 S3 VF 最后一次读取 |
| S3 update | `fp32(R_c)`与D及gate做FP32运算，末尾转BF16 | R、D、gate保持FP32运算 | 生成下一R和公开输出 |
| `final_state` | BF16 | FP32 | 算子输出 |

FwdH 没有 `beta` 输入、中间量或输出：

```text
beta dtype      = N/A
beta UB 占用    = 0 KiB
beta L1 占用    = 0 KiB
beta 生命周期   = N/A
```

### 2.2 符号

```text
B       batch；varlen 时物理 B=1
N       compact sequence 数
T       总 token 数
HK      key head 数
HV      value head 数
BT      chunk_size，固定 64
M       当前 chunk 的有效 token 数，1 <= M <= BT
L_n     sequence n 的有效长度；dense时为T，varlen时为cu_seqlens[n+1]-cu_seqlens[n]
Nc_n    sequence n 的chunk数，ceil(L_n/BT)
Ctot    所有sequence的物理chunk槽总数；dense单个batch轴内为Nc_n，varlen为sum_n Nc_n
tokenBase(n,c)  dense为c*BT；varlen为cu_seqlens[n]+c*BT
chunkPrefix(n)  sum_{j<n} Nc_j
globalChunkId(n,c)  dense为c；varlen为chunkPrefix(n)+c
K       key/state 行维，固定 128
V       value/state 列维，固定 128
c       当前 sequence 内 chunk 编号
hv      value head 编号
hk      g-only 时 hv 映射到的 raw-key head 编号
kh      Stage 2 物理左操作数 head：g-only 时 kh=hk，gk-only 时 kh=hv
E(x)    use_exp2=false 时为 exp(x)，true 时为 exp2(x)
g_last  当前 chunk 最后一个有效 token 的 scalar gate，即 g_c[M-1]
gk_last 当前 chunk 最后一个有效 token 的 key-wise gate，即 gk_c[M-1,:]
R_c     StateT 的 chunk c 递推状态；S3读取它并生成R_{c+1}，[K,V]
H_c     BF16 的Cube输入shadow，也是公开h的chunk c元素；不回读参与S3递推，[K,V]
Pacc_c  Stage 0 的 FP32 L0C 累加结果 W_c @ H_c，[M,V]
P_c     cast_PType(Pacc_c)，Stage 0 到 Stage 1 的中间量，[M,V]
V_new_c 公开 `v_new`，BF16 [M,V]
V_new_g_c 仅 g-only 存在的 Stage 2 BF16 右操作数，[M,V]
kg_c     Stage 2 的逻辑左操作数；g-only 为 `E(g_last-g_i) * k_raw_c[i,:]`，
         gk-only 为已由 Prepare 得到的 `kg_c`，[M,K]；g-only 可用 `k_raw_c^T @ V_new_g_c` 形式计算
Dacc_c  Stage 2 的 FP32 L0C 累加结果，[K,V]
D_c     Dacc_c，Stage 2 到 Stage 3 的 FP32 中间量，[K,V]
k_raw_c g-only 的 raw key 输入；当写入本轮 kg slot 时，是该 slot 的物理 payload，[M,K]
state_gm_offset(base,k,v,state_v_first)
        = base + (state_v_first ? v * K + k : k * V + v)
        # base 为元素基址，返回 logical state[k,v] 的 GM 元素偏移；h/initial_state/final_state 共用该规则
```

`R_c` 的 dtype 是 Host 按 3.1 节优先级推导出的 `StateT`；`DTYPE_FINAL_STATE` 只承载已经
完成推导的编译分派结果，不能在 `final_state=nullptr` 时作为 dtype 来源。`H_c` 必须从已经
量化到 `StateT` 的 `R_c` 转换得到，不能从量化前 FP32 临时结果直接生成。`StateT=BF16`
时 `H_c` 与 `R_c` 数值和物理 dtype 相同；`StateT=FP32` 时 `H_c` 是 FP32递推state 的
BF16 Cube shadow。

### 2.3 主要张量与单 head 容量

| 张量 | Shape | dtype | 单 head 大小 | 用途 |
| --- | --- | --- | ---: | --- |
| `w_c` | `[BT,K]` | BF16 | 16 KiB | Stage 0 左操作数 |
| `H_c` | `[K,V]` | BF16 | 32 KiB | Stage 0 右操作数、公开 `h` |
| `P_c` | `[BT,V]` | BF16/FP32 | 16/32 KiB | S0 输出；保留到对应 S1 VF 最后一次读取；dtype=PType |
| `u_c` | `[BT,V]` | BF16 | 16 KiB | Stage 1 输入 |
| `V_new_c` | `[BT,V]` | BF16 | 16 KiB | 公开 `v_new`；gk-only 的 S2 右操作数 |
| `V_new_g_c` | `[BT,V]` | BF16 | 16 KiB | g-only 的 S1 输出到 L1 zN；保留到对应 S2 MTE1 最后一次读取 |
| `k_raw_c` | `[BT,K]` | BF16 | 16 KiB | g-only输入 `k` 的当前chunk视图；物理kg slot的payload |
| `kg_c` | `[BT,K]` | BF16 | 16 KiB | Stage 2逻辑左操作数；gk-only时也是Prepare后的物理输入 |
| `D_c` | `[K,V]` | FP32 | 64 KiB | S2 输出；保留到对应 S3 VF 最后一次读取；dtype=DType |
| `R_c` | `[K,V]` | BF16 | 32 KiB | BF16 rolling state；保留到 S3 并原位生成下一 chunk state |
| `R_c` | `[K,V]` | FP32 | 64 KiB | FP32 rolling state；按容量 fallback 在 GM 跨 chunk 保留 |
| `g_c` | `[BT]` | BF16 | 0.125 KiB | g-only gate，单 head |
| `g_c` | `[BT]` | FP32 | 0.25 KiB | g-only gate，单 head |
| `gk_last` | `[K]` | BF16 | 0.25 KiB | gk-only 最后一行，单 head |
| `gk_last` | `[K]` | FP32 | 0.5 KiB | gk-only 最后一行，单 head |

每个 AIV 最多同时管理两个 head，因此：

```text
P[2], StateT=BF16  = 32 KiB BF16
P[2], StateT=FP32  = 64 KiB FP32
D[2], StateT=BF16  = 128 KiB FP32
D[2], StateT=FP32  = 128 KiB FP32
R[2], StateT=BF16  = 64 KiB
R[2], StateT=FP32  = 128 KiB
g[2], GateT=BF16   = 0.25 KiB
g[2], GateT=FP32   = 0.5 KiB
gk_last[2], BF16   = 0.5 KiB
gk_last[2], FP32   = 1 KiB
```

### 2.4 主要张量

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `k` | g-only `[B,HK,T,K]`；gk-only `[B,HV,T,K]` | g-only 为 raw k；gk-only 为 Prepare 后按 value head展开的 kg |
| `w` | `[B,HV,T,K]` | Stage 0 左操作数 |
| `u` | `[B,HV,T,V]` | Stage 1 输入 |
| `g` | `[B,HV,T]` | 可选关键字输入；非空时选择 g-only |
| `gk` | `[B,HV,T,K]` | 可选关键字输入；非空时选择 gk-only |
| `initial_state` | `state_v_first=false` 为 `[N,HV,K,V]`；`true` 为 `[N,HV,V,K]` | BF16/FP32，可空 |
| `h` | `false`：dense `[B,HV,Nc_n,K,V]`、varlen `[1,HV,Ctot,K,V]`；`true`：dense `[B,HV,Nc_n,V,K]`、varlen `[1,HV,Ctot,V,K]` | BF16，每个 chunk 的起始状态；varlen按globalChunkId写入 |
| `v_new` | `[B,HV,T,V]` | BF16 |
| `final_state` | `state_v_first=false` 为 `[N,HV,K,V]`；`true` 为 `[N,HV,V,K]` | BF16/FP32，可空；非空时 dtype 即 StateT |

`g` 与 `gk` 各自都是可选输入，但一次调用必须且只能提供一个：允许 `g=None, gk!=None` 或
`g!=None, gk=None`，不允许二者同时为空或同时非空。要求 `HV % HK == 0`，Stage 2
cache key按映射后的 `hk` 建立；gk-only 的物理 `k` 已是 Prepare 输出 `kg`，Host 必须拦截
`k.shape[1] == HV`，cache key按 `hv` 建立，禁止不同 value head共享 `kg` entry。下文统一把
这两种物理 head编号记为 `kh`。

g-only 的 GVA 映射和每轮实际 key 需求定义如下。下文用户口径的 `H_k:H_v` 对应本文符号
`HK:HV`，表示 key/value head 数量比例，不是状态张量 `H_c` 的下标：

```text
groupSize = HV / HK
hv_to_hk(hv) = floor(hv / groupSize)
active_hv_round(r) = [4*r, min(4*(r+1), HV))
required_hk_round(r) = unique({hv_to_hk(hv) : hv in active_hv_round(r)})
Nkg_round(r) = |required_hk_round(r)|
```

`Nkg_round` 是当前 round 需要从 GM 取得的 `k_raw`/`kg` 份数，也是本轮有效的 `kg slot`
数量；它满足 `Nkg_round <= active_hv_count <= 4`，不按整个 sequence 的 `HK` 或 `HV`
预分配。每个 `hk` 在当前 round 只做一次 GM->L1，映射到该 `hk` 的多个 value head 共享
该 slot；进入下一 round 后旧 slot 失效，即使 `hk` 相同也重新从 GM 读取。

典型 GVA 比例如下（`H_{c,hv}` 表示各 value head 对应的状态 resident）：
下表中的 `H_k/H_v` 仅表示 key/value head 数，不是状态张量 `H_c`。

| `HK:HV`（示例规模） | 当前 round 的 `head_v` | 本轮 `required_hk` / `Nkg_round` | 本轮读取关系 |
| --- | ---: | ---: | --- |
| `1:3`（`HK=1,HV=3`） | 3 | 1 个 `hk` / 1 份 | 3 个 `H_{c,hv}` 分别读取；同一 `H_k`/`k_raw` 只读取 1 次 |
| `1:2`（`HK=2,HV=4`） | 4 | 2 个 `hk` / 2 份 | 4 个 `H_{c,hv}` 分别读取；两个 `H_k`/`k_raw` 各读取 1 次 |
| `1:6`（`HK=1,HV=6`） | round0 为 4，round1 为 2 | 每轮 1 个 `hk` / 1 份 | 同一组的后 2 个 `head_v` 位于下一 round；不跨 round 保留，下一 round 重新读取该 `H_k`/`k_raw` |

因此，GVA 下 round 可以只使用 3 个 value-head 槽；`kg` 槽数和 Stage2 的左操作数加载
必须以 `required_hk_round` 为准，而不是固定填满 4 个槽。

dense 模式下每个 batch 独立使用本 batch 的 token/chunk 轴。varlen 模式下物理 `B=1`，sequence
`n` 的第 `c` 个 chunk 使用：

```text
tokenBegin    = tokenBase(n,c)
M             = min(BT, cu_seqlens[n+1] - tokenBegin)
hChunkIndex   = globalChunkId(n,c)

k/w/u/g/gk/v_new 的有效token范围 = [tokenBegin, tokenBegin + M)
h 的物理chunk槽                  = h[0, hv, hChunkIndex, :, :]，末两轴按 state_v_first 解释
initial/final_state 的sequence槽 = state[n, hv, :, :]，末两轴按 state_v_first 解释
```

varlen 的 `chunk_indices` 必须包含 `Ctot` 个 `[n,c]` pair，并与 `cu_seqlens` 推导出的
`chunkPrefix/globalChunkId` 一一对应；Host 拦截缺失、重复、越界或顺序不一致的 pair。kernel
不得直接用局部 `c` 作为 `h` 的物理 chunk 下标，也不得按 BT 读取尾部 GM padding。

`state_v_first` 是 FwdH 的原生布局属性，由 Host 透传到 tiling 和 kernel；逻辑 state 在计算内部
统一视为 `[K,V]`，只有 GM 的输入/输出物理末两轴按该属性选择：

```text
state_v_first=false:
    initial_state, h, final_state 的 GM 物理末两轴为 [K,V]
    logical state[k,v] 的线性地址 = base + k * V + v

state_v_first=true:
    initial_state, h, final_state 的 GM 物理末两轴为 [V,K]
    logical state[k,v] 的线性地址 = base + v * K + k
```

本文固定 `K=V=128`，因此两种模式的 shape 数字相同，但 `[k,v]` 的内存顺序不同；不能
因为 shape 相同就省略 `state_v_first`，该属性必须参与 tiling 和 kernel 的每次 state GM 读写。

kernel 内的布局处理边界固定为：

```text
S0 BF16 initial：按 state_v_first 的 GM 地址规则读入，并在 L1 形成 canonical H_c[K,V]
S-1 FP32 initial：按 state_v_first 的 GM 地址规则读入，VF 内形成 canonical H_0[K,V]；
                  h_0 GM 写回使用同一 state_v_first 规则，L1 resident 仍保持 H_0[K,V]
S3：             内部生成 canonical Rnext/Hnext[K,V]；h/final_state GM 写回按该规则寻址
```

上述读写使用 kernel 内目标 CANN 支持的 layout-aware/transpose 搬运或等价的固定 tile
访存路径，不新增 Stage、不新增独立 VF，也不依赖外部 L2 transpose。UB/L1 resident 始终
保持 canonical `[K,V]`，不做 UB/L1 compact、搬移或跨 round 布局切换。

## 3. 算子接口

```python
h, v_new, final_state = chunk_fwd_h(
    k,
    w,
    u,
    *,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    state_v_first=False,
)
```

### 3.1 输入和属性约束

| 名称 | dtype | 目标设计约束 |
| --- | --- | --- |
| `k/w/u` | BF16 | 三者只支持 BF16，K=V=128；g-only k.head=HK且HV%HK=0，gk-only k.head=HV |
| `g` | BF16/FP32 | 可选关键字参数；与 `gk` 恰好一个非空 |
| `gk` | BF16/FP32 | 可选关键字参数；与 `g` 恰好一个非空 |
| `initial_state` | BF16/FP32 | 可空；非空时其 dtype 优先决定 StateT |
| `final_state` | BF16/FP32 | 物理输出可空；无 initial_state 且非空时，其 dtype 决定 StateT |
| `output_final_state` | bool | false 时物理 final_state 为 `nullptr`，Python 返回 None；true 时提供物理 final_state |
| `use_exp2` | bool | false 为 exp，true 为 exp2 |
| `state_v_first` | bool | 原生支持 false/true；false 为 state `[K,V]`，true 为 state `[V,K]`，Host 与 kernel 使用同一布局属性 |
| `save_new_value` | bool | Host 仅接受 True |

Host 必须先校验 `output_final_state == (final_state != nullptr)`；属性与物理指针 presence 不一致
时直接返回参数错误，不进入 dtype 推导。`state_v_first` 为 `false` 时，所有非空 state 的末两轴
必须是 `[K,V]`；为 `true` 时必须是 `[V,K]`，`initial_state` 与 `final_state` 的物理布局必须
一致。Host 将该属性原样写入 tiling，kernel 直接按对应 GM 地址规则读写，不在调用前后分配
转置临时 Tensor。校验通过后，StateT 按以下优先级确定：

```text
initial_state != nullptr:
    StateT = dtype(initial_state)
    final_state非空时，其dtype必须与initial_state一致

initial_state == nullptr && final_state != nullptr:
    StateT = dtype(final_state)

initial_state == nullptr && final_state == nullptr:
    StateT = FP32
```

因此，当 `initial_state=None` 且 `output_final_state=false` 时，物理 `final_state=nullptr`，递推
固定使用 FP32。所有 dtype、shape、模式组合均由 Host 拦截；kernel 只实例化 Host 已选择的
合法模板。

无 initial_state 时，StateT 分支的可达组合为：

```text
output_final_state=true,  final_state=BF16 -> StateT=BF16
output_final_state=true,  final_state=FP32 -> StateT=FP32
output_final_state=false, final_state=null -> StateT=FP32
```

公开 Python 签名没有独立的 state dtype 选择参数，因此无 initial_state 时，adapter 默认以
FP32 分配非空 final_state；无 initial_state 的 BF16 final_state 分支由可预分配物理输出的
aclnn/直调接口选择。两条路径仍统一遵守“StateT 等于实际 final_state dtype”。

## 4. Stage 0--3 完整数学语义

本节中的地址均为 KiB 半开区间。`[2]` 表示当前 AIV 的两个 head slot；`[4]` 表示配对
AIC 的四个 head slot。每个 Stage 的搬运覆盖该 head 的完整 Tensor；尾 chunk 在同一次
VF/MMAD 内通过有效行 mask 和补零处理，不增加 Tail VF 或 Tail pass。

Stage 编号表示 round 内的逻辑 phase；一个 phase 可以包含最多 4 个互相独立的单-head
Stage task。每个 task 只搬一个 head 并调用一次完整 VF/MMAD。UB 中两个 64 KiB 的
local data slot 分别永久归属本 AIV 的 local slot 0/1。各 Stage 的空间图只标记当前
owner：S0 的 owner 是 P，S1 使用互不重叠的 P 输入区和 `V_new_g` 输出区，S2 的 owner
是 D。只有前一 owner 的末次消费和相关异步搬运全部完成、slot 归还 free 后，下一 Stage
才能在相同物理地址写入新语义。不同 local slot 永不别名，因此允许跨 head wavefront。

所有 UB/L1 tensor 在 Stage 开始前即确定固定 base 和最大保留区间。派生输出写入预先分配的
独立目标区，或同一次VF中已经完成最后读取的固定子区；不通过 dtype 收窄搬移尚未读取的
活数据。本文中的“slot 改变语义”只表示前一
语义已经到达末次消费者后，后续 Stage 直接在该固定区间产生新语义，不表示 UB/L1 copy、
compact 或 defragment。

首 chunk 按 initial-state presence 和 dtype 在 Stage 外分派；不存在通用 Sinit：

```text
initial_state=BF16:
    S0 Cube 直接读取 BF16 initial_state
      -> S1 Vector -> S2 Cube -> S3 Vector

initial_state=FP32:
    S-1 Vector SinitCastFP32ToBF16（仅执行一次）
      -> S0 Cube -> S1 Vector -> S2 Cube -> S3 Vector

initial_state=None:
    S0 skipped
    S1 Vector(no-P) -> S2 Cube -> S3 Vector
```

BF16 初态不经过独立 Vector 转换：AIC 在首个 S0 内按 `state_v_first` 的 GM 地址规则直接执行
`initial_state GM -> L1 BF16[K,V] -> L0B -> MMAD`。同一个 initial state 后续还要被 Vector 用于
公开 `h_0` 和 rolling-state 递推时，S1 可以按规则 10 独立执行一次 `GM -> UB`；Cube 与
Vector 各自读取同一份 GM 输入是明确允许的，不构成重复搬运违规。

FP32 初态不能直接参与 `BF16 w @ FP32 state`：A5 MMAD 不支持该混合类型，GM->L1 MTE2
也不承担 FP32->BF16 转换。因此仅该 dtype 分支增加一次性 `SinitCastFP32ToBF16`，生成
首个 S0 的 BF16 Cube shadow；实际参与后续S3递推的R仍为FP32。后续 S3 在同一次 VF 中同时
生成 FP32 `R_{c+1}` 和 BF16 `H_{c+1}`，不再增加转换 Stage。

这同时明确 FP32 state 的 Cube 数值契约：S0 计算
`BF16(w) @ cast_BF16(R_c)` 并以 FP32 累加，而不是 `fp32(w) @ R_c(FP32)`。CPU golden、
测试标杆和后续实现必须采用同一 chunk-boundary BF16 shadow 语义；否则就必须另行设计
W 的 FP32 转换和 FP32 MMAD 路径，不能把两种语义混用。

### S-1：Vector，仅为 FP32 initial_state 构造首个 Stage0 的 BF16 右操作数

数学语义：

```text
R_0 = initial_state                              # logical FP32 [K,V]，GM末两轴按 state_v_first
H_0 = cast_BF16(R_0)                            # canonical BF16 [K,V]
```

每个单-head S-1 task 一次搬入完整 FP32 initial state，并只调用一次 RegBase VF。当前 AIV
为两个 local head 分配两套独立输入/输出 bank：

```text
UB[0,64)     H_0[2]，BF16 [2,K,V]
               local slot 0: [0,32)
               local slot 1: [32,64)
             # 两个固定派生输出bank；各自写h GM和对应L1 resident后释放

UB[64,192)   initial_state[2]，FP32 [2,K,V]
               local slot 0: [64,128)
               local slot 1: [128,192)
             # 两个固定输入bank；来源GM，各自只供对应VF，VF最后读取后释放

UB[192,248)  空闲
```

单 task仍只处理一个head的64 KiB FP32输入和32 KiB BF16输出；整个S-1 phase的两套bank
合计有效数据192 KiB，地址高水位192 KiB。两个local head不再共享同一个H或initial scratch，
可以按bank执行MTE2/VF/MTE3 ping-pong。
S-1 的 initial GM MTE2 按 `state_gm_offset` 读取 logical `[K,V]`；`state_v_first=true` 时
输入物理为 `[V,K]`，不先生成外部转置副本。VF 输出的 canonical `H_0[K,V]` 一路写 L1，
另一路按同一地址规则直接写公开 `h_0`。
S-1 不把 `StateT` 从FP32改成BF16，也不把递推状态改写成BF16；它只为BF16 `tl.dot`生成
一次性的右操作数H0，并生成同值的公开h0。
所有 head 走相同路径，不把一部分初态驻留 UB、另一部分回写 GM。VF 完成后，H 通过 MTE3
同时写概念上的公开 `h_0` 和对应 L1 head resident slot；公开 `h_0` 的每个 logical
`[k,v]` 元素按 `state_gm_offset` 写入，dense 基址为 `h[n,hv,0,:,:]`，varlen 基址为
`h[0,hv,globalChunkId(n,0),:,:]`。由于S-1两套bank与main union存在
跨local-slot地址复用，单个head的MTE3完成时只记录本地done，不能立即向AIC发布可启动S0的
`H ready`。只有当前head round所有active initial bank都已被VF最后读取、所有active H bank的
GM/L1 MTE3都已完成并drain后，才按active mask统一发布各head的phase-gated `H ready`，并让
AIV进入main chunk loop。AIC在收到该phase-gated ready前禁止为任何head启动S0/Fixpipe。
这里不是把 BF16 shadow 写入 hidden GM workspace：H只在UB中临时生成，随后写公开h和供S0
消费的L1 resident。phase drain后才把`[0,192)`交给主循环按后续Stage语义复用；这不是
无等待地重置owner，也不需要`SyncAll`。
FP32 initial 不在这些bank跨 Stage 保留：首个 S3 若需要递推R0，按 FP32
rolling-state GM fallback 再读取原始 initial state。两次 Vector GM 读取是 `R[2]+D[2]`
无法同时容纳于 248 KiB 后按规则 14 选择的统一 fallback，不按 head 分流。

BF16 initial 和无 initial 两个分支均不创建 S-1 task。无 initial 时也不创建首个 S0 task；
S1 选择 no-P VF，并把 `P=0` 作为编译期语义处理。

### Stage 0：Cube，计算 P = W @ H

数学语义：

```text
Pacc_c = w_c @ H_c                              # FP32 [M,V]，L0C
P_c    = cast_PType(Pacc_c)                     # BF16/FP32 [M,V]，写UB
```

首 chunk 且 `initial_state=None` 时，Stage 0 完全跳过。Stage 1 选择无 P 的 VF，不能伪造
一次 `W @ 0`。

#### L1 空间布局

```text
L1[0,64)      W_c bank[4]
              # BF16；4 个固定 16 KiB zN 槽，分别对应本 round 的 4 个 value head
              # 每槽是一份完整 w_c[M,K]（M<=64,K=128）；仅供对应 S0 的 MMAD
              # 每槽有独立 MTE1 free/ready；当前 round 消费完后即可复用

L1[64,128)    Stage0 未占用

L1[128,256)   H_c resident[4]
              # BF16 [4,K,V]，每份 32 KiB
              # 首chunk BF16初态由本S0直接从GM填入；FP32初态来自S-1；后续来自S3
              # 保留到各自 S0 MTE1 消费完成

L1[256,320)   kg slot[4]
              # 可在 S0 期间异步预取当前 round 的 required_hk_round，共 Nkg_round 份（1 <= Nkg_round <= 4）
              # 同一 hk 只建立一个 slot；本 S0 不消费，持续保留到 S2 对应 slot 的最后一个本轮 MTE1 消费完成
              # g-only payload 为对应 hk 的 k_raw；gk-only payload 为对应 hv 的 Prepare 后 kg
              # 仅服务当前 round，禁止跨 round 保留
```

这里的 Stage 实例是单个 `(sequence, chunk, value-head)`，但一个 round 的四个
Stage0 task 各自拥有 `W_c bank[4]` 中的一个固定槽。`W_c[4]` 表示四份独立的
完整矩阵，不是把四份矩阵拼成一个逻辑矩阵；`L1[64,128)` 没有数据所有者，
不参与任何隐式搬运或位置拼接。`kg slot` 是 S0 期间真实发起 MTE2 后的当前 owner，
不是对未来语义的预标记；它与 W/H 地址不重叠，并保留到 S2 消费。每个 W/H 槽只保留到
对应 S0 的 MTE1 末次读取完成；归还 free 后，后续 Stage 才能在同一物理地址写入自己的数据。

首 chunk BF16 initial 由本 S0 直接从 GM 搬入对应 H resident slot，不等待 AIV producer；
首 chunk FP32 initial 等待 S-1 phase drain后统一发布的 `H ready`；后续 chunk等待上一 S3
的 `H ready`。`w_c`
对当前 head 完整搬入一次，H 在同一 S0 内只进入 L1 一次。

#### Cube 到 Vector 的 UB 输出

StateT=BF16 且 rolling state 常驻时：

```text
UB[0,16)      head 0 P_c，BF16 [BT,V]
UB[16,64)     head 0 Stage0 未占用
UB[64,80)     head 1 P_c，BF16 [BT,V]
UB[80,128)    head 1 Stage0 未占用
               # 两个 P 分别保留到各自 S1；不同 head 地址不重叠

UB[128,192)   R_c[2] BF16，S0 期间仍为 live data，本 S0 不读写
               # 首chunk BF16 initial时尚未装入，首个S1才从GM搬入
               # 后续chunk已由前一S3驻留，保留到当前S3
UB[192,248)   空闲
```

StateT=FP32 GM fallback 时：

```text
UB[0,32)      head 0 P_c，FP32 [BT,V]
UB[32,64)     head 0 Stage0 未占用
UB[64,96)     head 1 P_c，FP32 [BT,V]
UB[96,128)    head 1 Stage0 未占用
UB[128,248)   空闲
```

Stage 0 自身没有 Vector 指令。P 使用配对 AIV 的两份 UB slot；Fixpipe 完成后发布 `P ready`。

#### 操作流程

1. 若当前 round 后续存在 S2，scheduler 先根据 `active_hv_round` 计算 `required_hk_round`
   和 `Nkg_round`，再在 S0 期间为这些 distinct `(chunk,kh)` 异步预取恰好 `Nkg_round` 份
   `kg`/`k_raw`；同一 `hk` 只发起一次 GM->L1。S0 不等待也不消费 `kg`，只记录各 slot 的
   ready 代际。随后，首 chunk BF16 initial 分支由本 S0 先把完整 initial state 从 GM 搬入 H slot；FP32 initial
   和后续 chunk 分支等待对应 H slot ready。随后 MTE2 只搬入 w 的 `M` 个有效行。
   tail 场景在同一 Cube 搬入阶段用 MTE2 `InitConstValue` 清零 L1A 后再覆盖有效行，因而
   `w[M,AlignUp(M,16))` 明确为零；不越界读取 GM，也不引入 Vector 指令或额外 Stage。
2. MTE1 将完整 w/H 搬入 L0A/L0B。
3. 执行一个逻辑 `w @ H`；tail 的 w 已在 L1 补零到 `AlignUp(M,16)`，MMAD 不读取
   GM padding，也不会让 L1 旧值参与计算。
4. Fixpipe 按 `PType` 将完整 P 直接写配对 AIV UB，不经过 GM：StateT=BF16 使用
   `F322BF16`，StateT=FP32 使用 `NoQuant`。
5. 发布 `P ready`；P 保留到对应 S1 VF 的最后一次读取。本 Stage 的任何 MMAD 都不读取
   P 或其他本 Stage 输出。

### Stage 1：Vector，计算 v_new 和 Stage 2 输入

数学语义：

```text
V_new_fp32 = fp32(u_c) - fp32(P_c)             # FP32 [M,V]
V_new_c    = cast_BF16(V_new_fp32)             # BF16 [M,V]；公开 v_new

g-only:
    g_last    = g_c[M-1]                        # 最后一个有效 token
    V_new_g_c[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32[i,:])
    alpha_c   = E(g_last)                       # FP32 scalar

gk-only:
    Stage2 直接使用 V_new_c，不生成 V_new_g_c
```

`V_new_c` 写公开 `v_new`。g-only 的 `V_new_g_c` 在 UB 中保持 ND，MTE3 按 zN 物理地址直接写对应 AIC 的
L1 resident slot；不落 GM。g-only 的 `alpha_c` 在同一次 VF 中生成并保留到 S3，避免重复
读取 g 或重复计算指数。

#### L1 当前 owner 布局

S1 只能在对应 S0 的 `H_c` 已被 MTE1 末次读取并归还 free 后写入这些地址。空间图按 gate
分支只标记本 Stage 实际产生的 owner：

```text
g-only:
L1[128,160)   head slot 0：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[160,192)   head slot 1：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[192,224)   head slot 2：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[224,256)   head slot 3：V_new_g_c，BF16 zN，前 16 KiB 有效

gk-only:
L1[128,160)   head slot 0：V_new_c，BF16 zN，前 16 KiB 有效
L1[160,192)   head slot 1：V_new_c，BF16 zN，前 16 KiB 有效
L1[192,224)   head slot 2：V_new_c，BF16 zN，前 16 KiB 有效
L1[224,256)   head slot 3：V_new_c，BF16 zN，前 16 KiB 有效

```

以上 owner 保留到对应 S2 的 MTE1 末次读取完成。S1 不在该区同时保留 `H_c`，也不在
这里预标记 S3 才会产生的 `H_{c+1}`。若 `kg` 已由 S0 预取，它在 S1 期间继续占用
`L1[256,320)`，但不属于 S1 当前 owner；本图不重复列出，生命周期仍闭合到 S2 最后
一个本轮 MTE1 消费完成。

#### StateT=BF16，g-only UB 布局

```text
UB[0,16)       local slot 0 P_c，BF16 [BT,V]
UB[32,48)      local slot 0 V_new_g_c，BF16 [BT,V]
UB[64,80)      local slot 1 P_c，BF16 [BT,V]
UB[96,112)     local slot 1 V_new_g_c，BF16 [BT,V]
               # P 与 V_new_g 同时存在于预分配的两个独立连续区，VF不做压紧
               # V_new_g 保留到 MTE3 写 L1 resident 完成

UB[128,192)    R_c[2]
               # BF16；[2,K,V]；首个S1从BF16 initial搬入/置零，或来自上一S3
               # 本 S1 不改，保留到 S3 以及下一 chunk

UB[192,224)    V_new work bank[2]：进入S1时owner为u_c，逐元素末次读取后owner为V_new_c
               # BF16；[2,BT,V]；来源 u GM
               # 同一次 VF 原位更新；v_new GM MTE3 完成后释放

UB[224,224.5)  head 0 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
UB[224.5,225)  head 1 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
               # 来源 g GM；仅本 head S1 VF，VF 完成后释放

UB[225,225.5)  head 0 alpha_c，FP32，仅 4 Byte 有效
UB[225.5,226)  head 1 alpha_c，FP32，仅 4 Byte 有效
               # 本 S1 VF 生成，保留到对应 head S3 完成

UB[226,248)    空闲
```

BF16/FP32 gate 的地址峰值均为 226 KiB，小于 248 KiB。

#### StateT=BF16，gk-only UB 布局

```text
UB[0,16)       head 0 P_c，BF16；本 S1 VF 完成后释放
UB[64,80)      head 1 P_c，BF16；本 S1 VF 完成后释放
UB[128,192)    R_c[2] BF16；保留到 S3/下一 chunk
UB[192,224)    V_new work bank[2]：进入S1时owner为u_c，逐元素末次读取后owner为V_new_c，BF16
               # 同一 ND 结果分别写 v_new GM 和 L1 resident；本分支不生成 V_new_g
UB[224,248)    空闲
```

gk-only 的 Stage 1 不读取 gk，因为 chunk 内相对衰减已经吸收到输入 `kg`。地址峰值 224 KiB。

#### StateT=FP32，g-only UB 布局

```text
UB[0,32)       local slot 0 P_c，FP32 [BT,V]
UB[32,48)      local slot 0 V_new_g_c，BF16 [BT,V]
UB[64,96)      local slot 1 P_c，FP32 [BT,V]
UB[96,112)     local slot 1 V_new_g_c，BF16 [BT,V]
               # P 与 V_new_g 使用独立固定区，VF不做UB内压紧
UB[128,160)    V_new work bank[2]：进入S1时owner为u_c，逐元素末次读取后owner为V_new_c，BF16
UB[160,224)    FP32 state scratch，S1 阶段空闲
UB[224,224.5)  head 0 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
UB[224.5,225)  head 1 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
UB[225,225.5)  head 0 alpha_c，FP32，仅 4 Byte 有效，保留到 S3
UB[225.5,226)  head 1 alpha_c，FP32，仅 4 Byte 有效，保留到 S3
UB[226,248)    空闲
```

FP32 rolling state 此时位于 GM，不占 S1 UB。gk-only 不使用 gate/alpha 区；两种模式地址峰值
均不超过 226 KiB。

#### 首 chunk无 initial_state 的无 P 分支

下述 BF16 分支只在 `output_final_state=true && dtype(final_state)=BF16` 时可达；FP32 分支在
`output_final_state=true && dtype(final_state)=FP32`，或
`output_final_state=false && final_state=nullptr` 时可达。

```text
StateT=BF16:
UB[128,192)    R_0[2]，BF16；同一次 no-P VF 置零
               # 同时作为公开H0的MTE3源；保留到首个S3
UB[32,48)      head 0 V_new_g_c，BF16 [BT,V]，仅 g-only
UB[96,112)     head 1 V_new_g_c，BF16 [BT,V]，仅 g-only
UB[192,224)    u_c[2] -> V_new_c[2]，BF16

StateT=FP32:
UB[0,32)       head 0 H_0，BF16 [K,V]
UB[32,48)      head 0 V_new_g_c，BF16 [BT,V]，仅 g-only
UB[64,96)      head 1 H_0，BF16 [K,V]
UB[96,112)     head 1 V_new_g_c，BF16 [BT,V]，仅 g-only
               # H0 与 V_new_g 由同一次S1 VF产生；两路MTE3完成后local data slot才可写D
UB[128,160)    V_new work bank[2]：进入S1时owner为u_c，逐元素末次读取后owner为V_new_c，BF16
UB[160,224)    当前 head FP32 state scratch，首个 S3 由 zero-state VF 写入

UB[224,224.5)  head 0 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
UB[224.5,225)  head 1 g_c；BF16 有效 0.125 KiB，FP32 有效 0.25 KiB
               # g-only 仅用于生成 V_new_g；R_0=0 时首个 S3 不需要 alpha
               # gk-only 本 S1 不使用该区
```

g-only 的 gate 分片按固定半 KiB slot 预留，因此地址高水位为 225 KiB；gk-only 的活跃
地址高水位为 224 KiB。同一次 VF 完成 R0/H0 置零、V_new 和 V_new_g，不增加单独初始化
Vector Stage。R0/H0 写公开 `h_0` 时，logical `[k,v]` 元素统一按 `state_gm_offset` 寻址；
FP32 首个 S3 不从 rolling GM 读数据；BF16 首个 S3 不使用未初始化 state。

当 `StateT=BF16` 时，R0 本身就是 H0：H0 的 GM MTE3可直接以
`UB[128,192)` 为源，local data slot 内不需要 H0 区；g-only 的两个 local head分别使用
固定 V_new_g 目标区 `[32,48)` 和 `[96,112)`。gk-only 不生成 V_new_g，直接使用
V_new work区中的 V_new。StateT=FP32 的 zero-state H0才使用上图的独立固定目标区。

#### 最终 chunk且不输出 final_state 的 v_new-only 分支

当 `output_final_state=false` 且当前为最终 chunk，S2/S3 没有消费者。S1 外层选择独立的
`v_new-only` VF：只搬 u 和必要的 P，只计算/写 `V_new`，不搬 g/gk、不生成 `V_new_g`/alpha、不写 L1。
若该 sequence 只有一个 chunk：无 initial 时同一个 no-P VF另外写零H0；BF16 initial时
同一个with-P VF额外执行一次完整`initial_state GM -> UB`并写非零h0，但不保留无消费者的R0；
FP32 initial的h0已经由S-1写出，S1不重复读取initial。三种分支均不生成无消费者state。
若本分支消费了P但没有 `V_new_g`/S2 来接续，S1在VF最后读取P后按5.6节条件发布`unionFree`给下一
head round中实际存在的AIC S0；若下一round仍因无initial而跳过S0，则不向AIC发布token，
由下一次no-P S1的本核owner链闭环。首chunk无initial的FP32分支会在union生成H0，不能
把该union视为从未占用。

#### ND UB 到 zN L1 的 MTE3

VF 输出完整 ND `V_new_g_c[M,V]` 后，AIV MTE3 按 C0 列块直接散写 L1：

```text
src ND offset = row * V + colBlock * 16
dst zN offset = (colBlock * AlignUp(M,16) + row) * 16

blockCount = AlignUp(M,16)
blockLen   = 1                       # 一个 32 Byte C0
srcGap     = V/16 - 1 = 7            # 32 Byte block 单位
dstGap     = 0
```

V=128 时每 head 发起 8 个 C0 DataCopy 描述符，但这是一次完整 MTE3 搬出阶段：源 Tensor
只在 UB 出现一次，不重新 MTE2、不重新 VF，也不按行形成多个 pass。尾部 `[M,AlignUp(M,16))`
在同一次 VF 中置零，并由同一组描述符写入 L1，不能残留 resident slot 的旧数据。

#### 操作流程

1. 等待 P ready；无 initial 的首 chunk 选择 no-P VF，不等待不存在的 P。
2. MTE2 搬入当前 chunk 的有效 `u[0,M)` 前检查该 local V_new work bank 的 owner 状态。S-1 phase
   已完成并drain时该bank为`FREE`，不伪造等待；若上一owner是MTE3且下一owner为MTE2，
   则等待对应`uvToMte2Free[slot]`。g-only只搬有效`g[0,M)`，两者均不得按BT读取GM padding。
3. 外层按 `G/GK`、`exp/exp2`、`with-P/no-P`、`full/v_new-only` 选择不同 RegBase VF。
4. 每个 head 只调用一次 VF，VF 内完成全 M 行、V=128 全列运算和尾部补零。
5. g-only 从独立`V_new_g`区MTE3写L1，并从V_new work bank只写`V_new[0,M,0,V)`有效GM区域；gk-only
   的同一 V_new work bank
   同时作为 `V_new` GM与L1 `V_new` 的源，必须把两路MTE3都计入该bank的最后使用；v_new-only只计
   `V_new` GM。
6. bank 的最后一条 MTE3 发起后，由 scheduler 根据下一 owner 只发布实际会被消费的事件：
   下一owner是后续S1 MTE2时发布`uvToMte2Free[slot]`（`MTE3_MTE2`）；下一head round
   先执行S-1时，必须等待该bank真实MTE3完成并在round末terminal drain，S-1不能靠重置
   owner抢占；整个sequence没有后继时同样terminal drain。普通S3不再消费V_new work bank。P和实际存在的 gate
   临时区释放，BF16 state及 g-only alpha按声明保留。

g-only 尾 chunk和 varlen 的 `g_last` 始终按 `M-1` 定位，不能按固定 `BT-1` 读取 padding。

### Stage 2：Cube，计算 delta_h

数学语义：

```text
g-only:
    Dacc_c = kg_c^T @ V_new_c                  # FP32 [K,V]，L0C
            = k_raw_c^T @ V_new_g_c            # 等价实现；g-only kg slot 保存 k_raw

gk-only:
    Dacc_c = kg_c^T @ V_new_c                  # FP32 [K,V]，L0C
                                               # gk-only 的 kg_c 即 Prepare 后输入

D_c = Dacc_c                                    # FP32 [K,V]，NoQuant写UB
```

gk-only 的 `k` 形参已经是 Prepare 后的 `kg`，Stage 2 不再乘一次 gk。
`kg` 是数学上的门控 key 名称，不要求在 Stage0 预先物化成多份物理副本；当前物理
kg slot 只按本轮实际需求保存 raw `k`（g-only）或 Prepare 后的 `kg`（gk-only），门控等价关系由
`V_new_g` 或输入 `kg` 表达，并且仅在当前 round 内有效。

#### L1 空间布局

```text
g-only:
L1[128,160)   head slot 0：V_new_g_c，前 16 KiB 有效，BF16 zN
L1[160,192)   head slot 1：V_new_g_c，前 16 KiB 有效，BF16 zN
L1[192,224)   head slot 2：V_new_g_c，前 16 KiB 有效，BF16 zN
L1[224,256)   head slot 3：V_new_g_c，前 16 KiB 有效，BF16 zN

gk-only:
L1[128,160)   head slot 0：V_new_c，前 16 KiB 有效，BF16 zN
L1[160,192)   head slot 1：V_new_c，前 16 KiB 有效，BF16 zN
L1[192,224)   head slot 2：V_new_c，前 16 KiB 有效，BF16 zN
L1[224,256)   head slot 3：V_new_c，前 16 KiB 有效，BF16 zN

两种 gate 分支的当前 Stage 公共输入：
L1[256,272)   kg slot 0，BF16 [BT,K]
L1[272,288)   kg slot 1，BF16 [BT,K]
L1[288,304)   kg slot 2，BF16 [BT,K]
L1[304,320)   kg slot 3，BF16 [BT,K]
              # 每 slot 16 KiB；g-only payload为对应hk的k_raw，gk-only payload为对应hv的kg
              # key=(chunk,kh)，只保存当前 round 的 required_hk_round，共 Nkg_round 个 distinct key
              # round 结束后全部失效，下一 round 不复用旧 kg 数据
```

`L1[0,128)` 在 Stage2 不分配 owner，因此本图不再重复画 Stage0 的 W。右操作数从 S1
保留到本 S2 的 MTE1 末次读取完成；`kg` 可在 S0 期间为当前 round 异步预取，未预取时
才在本 S2 消费前按需装载。两类输入
各自完成最后一次 MTE1 读取后立即归还 free，S3 才能在已释放的 head slot 地址写
`H_{c+1}`。

g-only 的 GVA/MQA 下，scheduler 在进入每个 round 时先计算 `required_hk_round`，再按
`Nkg_round` 为当前 `(chunk,kh)` 分配精确数量的 slot。同一 round 内映射到同一个 `hk` 的
多个 value head 共享同一只读 raw-k slot；各 head 的门控因子仍在 `V_new_g` 中独立计算。
gk-only 使用 `kh=hv`，每个 value head 的 `kg` 独立，禁止跨 value head 共享。每个 slot
只在当前 round 的最后一个 S2 MTE1 消费者完成后失效，下一 round 从 GM 重新装载，不统计
也不保留跨 round 的 key 总数。

原方案的 16 份不是算法要求，而是为“所有 key 跨 head round 驻留”预留的容量上限：
每份 16 KiB，16 份占 256 KiB。当前调度约束是一个 round 最多 4 个 active head，且不把
kg 带到下一 round，因此 `Nkg_round <= 4`；实际只为 `required_hk_round` 建立 slot。例如
`HK=1,HV=3` 时本轮只建立 1 份，`HK=2,HV=4` 时建立 2 份，`HK=1,HV=6` 时两个 round
各建立 1 份并在第二 round 重新读取。原方案多出的 12 份没有当前消费者，既不能减少本轮
的 MMAD，也不能在下一 round 使用，属于无效驻留，故从设计中删除。
若仍按 16 份物理预留，`kg` 单项就占 256 KiB；再加上当前 W 的 64 KiB 和 H/resident 的
128 KiB，至少需要 448 KiB，无法与本文固定的 320 KiB L1 地址上界同时成立。

#### Cube 到 Vector 的 UB 输出

StateT=BF16：

```text
UB[0,64)       head 0 D_c，FP32 [K,V]
UB[64,128)     head 1 D_c，FP32 [K,V]
               # S1 的 P 末次读取和相关 MTE3 完成、slot free 后，S2 Fixpipe 才能写 D
UB[128,192)    R_c[2] BF16；跨 Stage 保留，本 S2 不读写
UB[224,225)    两个 head 的 gate 临时分片，本 S2 空闲
UB[225,225.5)  head 0 alpha，FP32，仅 4 Byte 有效，g-only
UB[225.5,226)  head 1 alpha，FP32，仅 4 Byte 有效，g-only
               # 各自跨本 head S1->S3 保留
               # gk-only 不使用 gate/alpha 区
UB[226,248)    空闲
```

StateT=FP32：

```text
UB[0,64)       head 0 D_c，FP32 [K,V]
UB[64,128)     head 1 D_c，FP32 [K,V]
               # S1 的前序 owner 全部释放后，S2 Fixpipe 才能写 D
UB[128,160)    S1 V_new work区，S2 时已释放
UB[160,224)    FP32 state scratch，S2 时空闲
UB[224,225)    两个 head 的 gate 临时分片，本 S2 空闲
UB[225,225.5)  head 0 alpha，FP32，仅 4 Byte 有效，g-only
UB[225.5,226)  head 1 alpha，FP32，仅 4 Byte 有效，g-only
               # 各自跨本 head S1->S3 保留
               # gk-only 不使用 gate/alpha 区
UB[226,248)    空闲
```

Stage 2 没有 Vector 指令。D 使用配对 AIV 的两份完整 UB slot，不写 GM。

#### 操作流程

1. scheduler 先计算当前 `active_hv_round` 的 `required_hk_round` 和 `Nkg_round`，只为
   这些 `(chunk,kh)` 分配对应数量的 kg slot。若 S0 已异步预取，则按 `kh` 复用对应 slot
   及其 ready 代际；若 S0 被跳过或未预取，则在进入 S2 前只建立缺失的 slot。建立新 slot
   时先等 `kg overwrite-safe`，用 MTE2 `InitConstValue` 清零完整 16 KiB entry，再仅从
   GM 搬入 `M` 个有效行。MTE2 完成后由当前 round 的第一个 MTE1 消费者等待一次 ready
   event 并把 slot 标成 valid；同一 round 映射到该 `kh` 的后续 head 直接复用，不重复等待
   或设置同一个 token。这样 `kg[M,AlignUp(M,16))` 明确为零且不越界读取 GM。
2. 等待当前分支的右操作数 ready：g-only 等待 `V_new_g`，gk-only 等待 `V_new`。
3. MTE1 从 kg slot 和当前分支右操作数 resident 分别搬入 L0A/L0B。
4. 当前 round 映射到该 kg slot 的最后一个 head 的 MTE1 完成后归还 `kg overwrite-safe`；
   round 结束时所有 kg slot 均失效，下一 round 必须从 GM 重新装载，不保留旧数据。
5. 执行一个逻辑转置 MMAD；tail 的 kg 和当前分支右操作数都已补零到 `AlignUp(M,16)`。
6. 两条 StateT 分支都使用 `NoQuant`，Fixpipe 将完整 FP32 D 直接写配对 AIV UB；随后发布
   `D ready`。D 不经过 GM、不在 S2 末尾量化，并保留到对应 S3 VF 的最后一次读取。
7. 本 Stage 的任一 MMAD都不读取 D 或其他本 Stage 输出。

### Stage 3：Vector，更新 StateT rolling state

数学语义：

```text
g-only:
    RnextFp32 = fp32(R_c) * alpha_c + fp32(D_c)

gk-only:
    gk_last = gk_c[M-1,:]                       # 最后一个有效 token
    row_gate[r] = E(gk_last[r])
    RnextFp32[r,:] = fp32(R_c[r,:]) * row_gate[r] + fp32(D_c[r,:])

R_{c+1} = cast_StateT(RnextFp32)
H_{c+1} = cast_BF16(fp32(R_{c+1}))
```

先量化到 StateT，再从该量化结果生成 H。非末 chunk将 H 同时写概念上的公开 `h_{c+1}` 和
L1 head resident slot；公开 h 的每个 logical `[k,v]` 元素按 `state_gm_offset` 写入，dense
基址为`h[n,hv,c+1,:,:]`，varlen 基址为`h[0,hv,globalChunkId(n,c+1),:,:]`。最终 chunk不再生成下一 H。
`output_final_state=true` 时最终 R 写公开 `final_state`，同样按 `state_gm_offset` 写入；false 时最后一个
chunk 的 S2/S3 整体跳过。

#### L1 输出布局

仅非末 chunk 的 S3 产生下一块状态。对应 S2 的右操作数已被 MTE1 末次读取并归还 free 后，
S3 才能在相同 head slot 写入当前 Stage 的 owner：

```text
L1[128,160)   head slot 0：H_{c+1}，BF16 [K,V]，32 KiB
L1[160,192)   head slot 1：H_{c+1}，BF16 [K,V]，32 KiB
L1[192,224)   head slot 2：H_{c+1}，BF16 [K,V]，32 KiB
L1[224,256)   head slot 3：H_{c+1}，BF16 [K,V]，32 KiB
```

最终 chunk 不生成 `H_{c+1}`，因此不写上述 L1 区。该图只表示 S3 的 owner，不再同时
标记已在 S2 释放的 `V_new_g` 或 `V_new`。

#### StateT=BF16 的 UB 布局

```text
UB[0,64)       head 0 D_c，FP32 [K,V]
UB[64,128)     head 1 D_c，FP32 [K,V]
               # 来源 S2 Fixpipe；各 head VF 完成最后一次读取后，D 语义结束
               # 最后一次读取完成后才归还local data slot free

UB[128,192)    R_c[2] -> R_{c+1}[2]
               # BF16；[2,K,V]；首个S1从initial搬入/置零，或前一S3产生
               # 固定地址逐元素覆写：源元素进寄存器后由同一地址保存新state
               # 非末 chunk 保留到下一 S3；末 chunk写出后释放

UB[192,224)    S1 V_new work区，本 S3 空闲
               # Hnext直接以更新后的BF16 state slot为MTE3源

UB[224,224.5)  head 0 gk_last；BF16 有效 0.25 KiB，FP32 有效 0.5 KiB
UB[224.5,225)  head 1 gk_last；BF16 有效 0.25 KiB，FP32 有效 0.5 KiB
               # 来源 gk GM；本 head S3 一次搬入，VF 完成后释放

UB[225,225.5)  head 0 alpha，FP32，g-only，仅 4 Byte 有效
UB[225.5,226)  head 1 alpha，FP32，g-only，仅 4 Byte 有效
               # 来源 S1，对应 head S3 完成后释放

UB[226,248)    空闲
```

地址峰值最大 226 KiB，小于 248 KiB。

#### StateT=FP32 的 UB 布局

```text
UB[0,64)       D_c head 0，FP32 [K,V]
UB[64,128)     D_c head 1，FP32 [K,V]
               # 两份D均从S2保留；D最后读取后语义结束

UB[128,160)    S1 V_new work bank[2]，本 S3 不读写
               # 与当前S3无数学依赖，可保持前序MTE3在途；不作为H scratch

UB[160,224)    当前 head 的 R_c -> R_{c+1}
               # FP32 [K,V]；普通递推来自 rolling GM，zero-state 由 VF 直接生成
               # 存在输入时一次完整 MTE2，每个 head 一次完整 VF
               # 仅 R_{c+1} 有后续 S3 消费者或需作为 final_state 输出时执行一次完整 MTE3
               # 固定地址逐元素覆写；当前head完成后由另一个head走同一路径
               # 整段使用独立state-scratch owner/event，禁止下一head提前覆写

UB[224,224.5)  head 0 gk_last；BF16 有效 0.25 KiB，FP32 有效 0.5 KiB
UB[224.5,225)  head 1 gk_last；BF16 有效 0.25 KiB，FP32 有效 0.5 KiB
UB[225,225.5)  head 0 alpha，FP32，g-only，仅 4 Byte 有效
UB[225.5,226)  head 1 alpha，FP32，g-only，仅 4 Byte 有效

UB[226,248)    空闲
```

进入 S3 时，上图中的 owner 只有 D。`H_{c+1}` 的固定派生目标是当前 head local data slot
的前 32 KiB：head0 使用 `[0,32)`，
head1 使用 `[64,96)`。同一次 VF 按 K 行正向处理；每行先把该行完整 FP32 D 读入寄存器，
确认该行 D 已完成最后读取后，该行地址才从 D owner 归还并改由 BF16 H owner 写入。
这个写入不会覆盖尚未读取的 D 行，不发生 UB
DataCopy、compact 或第二次 VF。地址峰值最大 226 KiB。两个 head 分别是两个 Stage 3 实例；
每个实例仍是完整 `[K,V]` 一次搬入和一次 VF，没有 K/V tile 或多 pass。

#### 操作流程

1. 等待当前 head 的 D ready。S3 不复用 `[128,160)` 的 V_new work bank，因此无需等待另一 head
   的 S1/v_new MTE3；无依赖的 S1 MTE3 与当前 S3 可以继续重叠。
2. BF16 state 直接使用跨 chunk UB resident。FP32 state 使用共享 `[160,224)` 前先检查其
   owner。S-1 phase 已经独立drain，不把V owner带入主循环；普通 rolling-state MTE2只按
   两态选择唯一动作：`FREE`直接写，`MTE3`等待`stateToMte2Free`（`MTE3_MTE2`）。首 chunk
   无 initial 的 zero-state VF 只需处理 `FREE` 直写或 `MTE3` 等待 `stateToVFree`
   （`MTE3_V`）。随后一次搬入完整 rolling GM slot，或由 zero-state VF 直接生成零输入；
   BF16 使用明确置零的 R0，FP32 不读取未初始化 rolling GM。
3. g-only 复用 S1 保存的 alpha；gk-only 只搬入数学上需要的完整 `gk_last[K]`。首个
   zero-state S3 不计算无效的 state gate；尾 chunk/varlen 按 `M-1` 定位 last token。
4. 外层按 gate 模式、exp 模式、StateT、state 来源和输出策略选择独立 RegBase VF。
5. 每个 head 只调用一次 VF，完整读取 D 和 state 并更新 StateT。StateT=BF16时更新后的state
   同时就是H；StateT=FP32时把BF16 H派生结果写入当前head local data slot的固定低32 KiB。
   D 的最后一次读取与H的派生写出都在该次VF内完成，不进行UB内搬运。
6. 非末 chunk将 local data slot低32 KiB中的 H（或 BF16 state）直接 MTE3 写 h GM 和对应 L1
resident slot；写 h GM 时每个 logical `[k,v]` 元素按 `state_gm_offset` 寻址；末 chunk不生成下一 H。
H两路MTE3完成前，该head local data slot不能被下一S0
   Fixpipe覆写。最终S3没有H ready可兼作union free：仅当下一实际写者是下一head round的
   AIC S0时，才在VF最后读取D后以PIPE_V发布独立`unionFree`；无initial使下一round跳过S0
   时不向AIC发token，由下一次S1/`V_new_g`-ready链闭环。最后round不set并terminal drain。V_new work bank只按自己的
   S1生命周期发布free，不再参与S3 H scratch事件。
7. 仅当 `R_{c+1}` 需要作为公开 final_state，或后续 S3 会再次读取它时，才发起 FP32
   rolling/final-state MTE3。发起后按下一 scratch owner只发布 `stateToVFree`（下一
   zero-state VF）或 `stateToMte2Free`（下一 rolling MTE2）；无后继则 terminal drain。
   若 `R_{c+1}` 没有状态消费者，则生成 H 后直接归还 state scratch，不发起 state MTE3，
   也不生成 state ready/event。发布实际存在的 H ready和state ready，并归还D与gate；不使用
   `SyncAll`。kernel/round退出时按每个bank/scratch的实际 owner、HardEvent类型和代际
   drain/release，不留下无消费者token。

## 5. Stage 资源分配方案

### 5.1 Stage 序列

BF16 initial state：

```text
S0    Cube   : BF16 initial_state -> L1，P = W @ H
S1    Vector : initial_state -> BF16 UB resident/h0；V_new、V_new_g、alpha
S2    Cube   : g-only D = kg^T @ V_new (= k^T @ V_new_g)；gk-only D = kg^T @ V_new
S3    Vector : Rnext、Hnext、final_state
```

FP32 initial state：

```text
S-1   Vector : SinitCastFP32ToBF16，生成BF16 h0/L1 Cube shadow
S0    Cube   : P = W @ H
S1    Vector : V_new、V_new_g、alpha
S2    Cube   : g-only D = kg^T @ V_new (= k^T @ V_new_g)；gk-only D = kg^T @ V_new
S3    Vector : 从原始FP32 initial/rolling GM读取R，生成Rnext/Hnext/final_state
```

两种 `state_v_first` 均走上述同一 Stage 链：`false` 的 GM state 为 `[K,V]`，`true` 的 GM
state 为 `[V,K]`；差异只体现在 S0/S-1 的 state 输入寻址和 S3 的 state 输出寻址，内部
L1/UB 仍使用 canonical `[K,V]`，不插入外部 L2 transpose 或额外 Stage。

无 initial state 的首 chunk：

```text
S0    skipped
S1    Vector : R0=0、H0=0、V_new、V_new_g；首个状态更新不保留无效 alpha
S2    Cube   : g-only D = kg^T @ V_new (= k^T @ V_new_g)；gk-only D = kg^T @ V_new
S3    Vector : zero-state VF，Rnext=D、Hnext/final_state
```

最终 chunk 且 `output_final_state=false`：

```text
chunk==0 && initial_state=None:
    S1(no-P, v_new-only)
    S0/S2/S3 skipped

其他情况:
    S0 -> S1(v_new-only)
    S2/S3 skipped
```

`v_new-only` S1 不搬 gate，不生成 `V_new_g`/alpha，不写 L1 resident。

Cube Stage 输入依赖检查：

```text
S0: P_hv <- W_hv, H_hv
    所有 W/H 都在进入 S0 前存在；任何 P 都不被本 S0 的 MMAD 消费。

S2: D_hv <- kg_hk, V_new_g_hv/V_new_hv
    kg 可在 S0 期间为当前 round 异步预取；S0 被跳过或未预取时在 S2 前按需装载；
    g-only 的 `V_new_g` 或 gk-only 的 `V_new`
    在完成 S1 后 ready；
    任何 D 都不被本 S2 的 MMAD 消费。
```

S1 在同一次 VF 中允许由`V_new_fp32`派生`V_new`、g-only的`V_new_g`和alpha的寄存器依赖；S3允许
`RnextFp32 -> StateT Rnext -> BF16 Hnext` 的寄存器依赖，符合 Vector Stage 规则。

### 5.2 L1 固定分配

同一物理地址可以跨 Stage 复用，但下列空间图只标记当前 Stage 的 owner，不把生命周期中的
多个变量写在同一行。地址移交必须经过“前一 owner 末次消费完成 -> wait/free 闭环 ->
后一 owner 写入”，不能仅靠 Stage 编号推定已经释放。

#### Stage0 当前 owner

```text
L1[0,64)       W_c bank[4]，4 x 16 KiB；每个 head 的完整 w_c，仅当前 S0 MMAD
L1[64,128)     Stage0 未占用
L1[128,160)    head slot 0：H_c，BF16 [K,V]
L1[160,192)    head slot 1：H_c，BF16 [K,V]
L1[192,224)    head slot 2：H_c，BF16 [K,V]
L1[224,256)    head slot 3：H_c，BF16 [K,V]
L1[256,272)    kg slot 0，BF16 [BT,K]
L1[272,288)    kg slot 1，BF16 [BT,K]
L1[288,304)    kg slot 2，BF16 [BT,K]
L1[304,320)    kg slot 3，BF16 [BT,K]
                  # S0 只为 required_hk_round 异步预取 Nkg_round 份；未使用槽保持 FREE
                  # 本 S0 不消费，已建立的 slot 保留到 S2 最后一个本轮消费者
```

W 和 H 分别保留到对应 S0 MTE1 的末次读取完成，随后归还 free。`kg` 与两者地址不重叠，
若在 S0 期间预取，则从 MTE2 完成起保持当前 owner，直到 S2 对应 slot 的最后一个本轮
MTE1 消费完成；不允许跨 round 保留。

#### Stage1 当前 owner

```text
g-only:
L1[128,160)    head slot 0：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[160,192)    head slot 1：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[192,224)    head slot 2：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[224,256)    head slot 3：V_new_g_c，BF16 zN，前 16 KiB 有效

gk-only:
L1[128,160)    head slot 0：V_new_c，BF16 zN，前 16 KiB 有效
L1[160,192)    head slot 1：V_new_c，BF16 zN，前 16 KiB 有效
L1[192,224)    head slot 2：V_new_c，BF16 zN，前 16 KiB 有效
L1[224,256)    head slot 3：V_new_c，BF16 zN，前 16 KiB 有效
```

S1 必须先确认对应 S0 的 H owner 已释放，再由 MTE3 写当前分支的唯一 owner。该 owner
保留到对应 S2 MTE1 的末次读取完成。若 `kg` 已由 S0 预取，它在 Stage1 期间继续占用
`L1[256,320)`，但不属于 S1 当前 owner；本图不重复列出，生命周期仍闭合到 S2 最后
一个本轮 MTE1 消费完成。

#### Stage2 当前 owner

```text
g-only:
L1[128,160)    head slot 0：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[160,192)    head slot 1：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[192,224)    head slot 2：V_new_g_c，BF16 zN，前 16 KiB 有效
L1[224,256)    head slot 3：V_new_g_c，BF16 zN，前 16 KiB 有效

gk-only:
L1[128,160)    head slot 0：V_new_c，BF16 zN，前 16 KiB 有效
L1[160,192)    head slot 1：V_new_c，BF16 zN，前 16 KiB 有效
L1[192,224)    head slot 2：V_new_c，BF16 zN，前 16 KiB 有效
L1[224,256)    head slot 3：V_new_c，BF16 zN，前 16 KiB 有效

两种分支的本 Stage 公共输入：
L1[256,272)    kg slot 0，BF16 [BT,K]
L1[272,288)    kg slot 1，BF16 [BT,K]
L1[288,304)    kg slot 2，BF16 [BT,K]
L1[304,320)    kg slot 3，BF16 [BT,K]
                  # 仅 required_hk_round 中的 Nkg_round 个 slot 有效，其余槽保持 FREE；不跨 round 保留
```

`kg` 可以来自 S0 对当前 round 的异步预取；S0 被跳过或没有预取时，最迟在 S2 消费前按
distinct `(chunk,kh)` 装载。右操作数和 `kg` 分别在各自最后一个本轮 MTE1 消费者完成后
释放；下一 round 不读取旧 slot。

#### Stage3 当前 owner

```text
L1[128,160)    head slot 0：H_{c+1}，BF16 [K,V]
L1[160,192)    head slot 1：H_{c+1}，BF16 [K,V]
L1[192,224)    head slot 2：H_{c+1}，BF16 [K,V]
L1[224,256)    head slot 3：H_{c+1}，BF16 [K,V]
```

S3 必须等待 S2 对应 head slot 的 MTE1 末次读取完成并归还 free 后才能写 H。H 保留到
下一 chunk S0 的 MTE1 末次读取；最终 chunk 不产生 H，因此该 Stage 不占用这些 L1 地址。

各 Stage 地址高水位分别为 S0 320 KiB（启用 `kg` 预取时）、S1 320 KiB（预取的 `kg`
仍在保留时）、S2 320 KiB、S3 256 KiB，物理总预留
高水位为 320 KiB。这里的 320 KiB 是固定地址上界，不表示四个 Stage 的数据同时存活。

### 5.3 UB dtype 分支与峰值

UB 允许随 Stage 改变语义，但只有 slot 的末次消费者完成并归还 free 后才能复用。两个
64 KiB local data slot 固定归属两个 local head：slot 0 使用 `[0,64)`，slot 1 使用
`[64,128)`。S0 写 P；S1 读取 P，并在独立子区产生 g-only 的 `V_new_g`；P 的末次读取以及
所有相关 MTE3 完成后，S2 才能把相同 local data slot 写为 D。S3 进入时该 slot 的 owner
仍是 D；FP32 state 分支只在某行 D 完成末次读取后，才把该行地址移交给 H。不同 local
slot 从不别名。

小 gate 区同样按 head 固定分片，避免跨 head wavefront 覆盖：

```text
UB[224,224.5)  local head 0 gate/gk_last，最大 0.5 KiB
UB[224.5,225)  local head 1 gate/gk_last，最大 0.5 KiB
UB[225,225.5)  local head 0 alpha，实际 4 Byte
UB[225.5,226)  local head 1 alpha，实际 4 Byte
```

#### StateT=BF16

| Stage | 两个 head 的活跃 UB 数据 | GateT=BF16 时有效数据总量 | GateT=FP32 时有效数据总量 | 地址高水位 | 保留关系 |
| --- | --- | ---: | ---: | ---: | --- |
| S0 | state 64 + P 32 | 96 KiB | 96 KiB | 192 KiB | P以BF16保留到S1 |
| S1 g | state 64 + P 32 + 独立V_new_g 32 + V_new work bank 32 + g + alpha | 约 160.26 KiB | 约 160.51 KiB | 226 KiB | work bank进入时owner为U，末次读取后原位写V_new；V_new_g写L1；alpha保留到S3 |
| S1 gk | state 64 + P 32 + V_new work bank 32 | 128 KiB | 128 KiB | 224 KiB | work bank进入时owner为U，末次读取后原位写V_new，写L1后释放 |
| S2 g | state 64 + D 128 + alpha | 约 192.01 KiB | 约 192.01 KiB | 226 KiB | D以FP32、alpha保留到S3 |
| S2 gk | state 64 + D 128 | 192 KiB | 192 KiB | 192 KiB | D以FP32保留到S3 |
| S3 g | state 64 + local data 128（进入S3为D）+ 允许异步在途的V_new work bank 32 + alpha | 约 224.01 KiB | 约 224.01 KiB | 226 KiB | D逐行末次读取后低32KiB地址才写H；state原位更新 |
| S3 gk | state 64 + local data 128（进入S3为D）+ 允许异步在途的V_new work bank 32 + gk_last | 224.5 KiB | 225 KiB | 225 KiB | D逐行末次读取后低32KiB地址才写H |

最坏有效数据峰值约 225 KiB；最高占用地址为 226 KiB，均小于 248 KiB。这里把与当前
S3 无依赖、仍可能在 MTE3 读取的另一 head V_new work bank计入同时活跃集合，因而容量结论不依赖
跨 head 执行顺序。BF16 rolling state 的两份 UB slot 跨 chunk 保留，不落 GM。

#### StateT=FP32

如果两份 state 和两份 D 同时驻留：

```text
R[2] FP32 = 128 KiB
D[2] FP32 = 128 KiB
合计       = 256 KiB > UB 248 KiB
```

尚未包含 gate 和输出，因此不能通过地址重排解决。为保持 delta_h 和 rolling state 的 FP32
精度，本设计按规则 14 使用 GM fallback：

```text
rolling_state_gm，逻辑 shape `[N,HV,K,V]`，StateT=FP32；hidden workspace 默认按 canonical `[K,V]` 存放

output_final_state=true:
    中间 chunk 可复用 final_state rolling 槽；若复用，读写按 state_v_first 的 GM 地址规则进行，最后一次写即公开输出

output_final_state=false:
    仅在存在后继 chunk、实际需要 rolling state 时使用 hidden rolling workspace
    单 chunk v_new-only 路径不生成 rolling state；Python 返回 None
```

| Stage | 两个 head 的活跃 UB 数据 | GateT=BF16 时有效数据总量 | GateT=FP32 时有效数据总量 | 地址高水位 | 保留关系 |
| --- | --- | ---: | ---: | ---: | --- |
| S-1 SinitCastFP32ToBF16 | initial state[2] 128 + H[2] 64 | 192 KiB | 192 KiB | 192 KiB | 两个独立bank；每个task仍只转换单head，phase drain后释放 |
| S0 | P 64 | 64 KiB | 64 KiB | 96 KiB | P 保留到 S1 |
| S1 g | P 64 + 独立V_new_g 32 + V_new work bank 32 + g + alpha | 约 128.26 KiB | 约 128.51 KiB | 226 KiB | work bank进入时owner为U，末次读取后原位写V_new；V_new_g写L1；alpha保留到S3 |
| S1 gk | P 64 + V_new work bank 32 | 96 KiB | 96 KiB | 160 KiB | work bank进入时owner为U，末次读取后原位写V_new，写L1后释放 |
| S2 g | D 128 + alpha | 约 128.01 KiB | 约 128.01 KiB | 226 KiB | D 保留到 S3 |
| S2 gk | D 128 | 128 KiB | 128 KiB | 128 KiB | D 保留到 S3 |
| S3 g | local data 128（进入S3为D）+ 允许异步在途的V_new work bank 32 + 当前state 64 + alpha | 约 224.01 KiB | 约 224.01 KiB | 226 KiB | D逐行末次读取后低32KiB地址才写H；state按消费者写GM |
| S3 gk | local data 128（进入S3为D）+ 允许异步在途的V_new work bank 32 + 当前state 64 + gk_last | 224.5 KiB | 225 KiB | 225 KiB | D逐行末次读取后低32KiB地址才写H |

GM fallback 不拆 Tensor：存在 rolling 输入时，每个 head 一次完整 64 KiB MTE2；每个
head一次完整 `[K,V]` VF；仅结果存在后续 S3 消费者或需输出 final_state 时执行一次完整
MTE3。有效数据峰值 225 KiB，最高占用地址为 226 KiB。它只牺牲 FP32 state 的本地跨
chunk 常驻，不改变计算精度。

如果把“一次 VF”解释为一次 VF 必须同时覆盖同一 AIV 的两个 head，则 FP32 state 路径无解；
GM 不能减少该 VF 的同时活跃输入集合。此时只能选择下列方案之一，并需另行批准：

1. 把 FP32 StateT 路径的 D 跨 Stage dtype 降为 BF16，改变数值精度；
2. 放宽两 head 同时进入一个 VF 的要求，采用本文的单 head Stage 实例；
3. 收窄 FP32 state 支持范围。

本文采用第 2 项，不静默降低 D 精度。

### 5.4 跨 Stage 数据保留原则

```text
Cube -> Vector:
    S0当前owner：P[2]由Fixpipe按PType直接写配对AIV UB；BF16路径32 KiB，FP32路径64 KiB
    P分别保留到S1末次读取，相关S1 MTE3完成后local data slot才归还free
    S2当前owner：D[2]在两条路径都由NoQuant Fixpipe写入FP32，共128 KiB
    D分别保留到S3末次读取；不同head local data slot永不重叠，均不经过GM

Vector -> Vector:
    BF16 R[2] 在 UB 跨 chunk 原位保留
    alpha[2] 从 S1 保留到 S3
    需要跨chunk/输出的FP32 R 因容量冲突按规则 14 使用 rolling GM

Vector -> Cube:
    S1 g-only：V_new_g[4] 由 AIV MTE3 直接写配对 AIC L1 zN resident
    S1 gk-only：V_new[4] 由 AIV MTE3 直接写配对 AIC L1 zN resident
    S3：Hnext[4] 在S2右操作数末次MTE1完成并free后写同一L1地址，同时写h GM
    下一 S0 直接消费该H，不再从GM读取

Cube -> Cube:
    当前数学链没有 Cube 输出直接供后续 Cube；kg 是 Stage2 的 GM 输入临时slot
    可在S0期间按required_hk_round异步预取当前round的kg/k_raw，并持续保留到S2最后一个本轮MTE1消费者
    每个当前 round/chunk 只保留Nkg_round个distinct (chunk,kh)；同一 round 内按 kh 共享
    当前 round 最后一个 S2 MTE1 消费者完成后立即失效，下一 round 从 GM 重新装载

公开输出:
    v_new 由 S1 写 GM
    h0由BF16-initial S1、FP32-initial S-1或no-initial S1按state_gm_offset写GM；
    Hnext由S3按同一规则写GM
    final_state 由最终 S3 按 StateT 和 state_gm_offset 写 GM
```

### 5.5 调度顺序

为了让 BF16 state 的两个 UB slot 和分Stage移交owner的四个 L1 head slot 跨 chunk 复用，目标 scheduler 的
循环顺序必须是：

```text
for sequence:
    for head_round:
        if head_round > 0:
            wait round_{head_round-1}.kgOverwriteSafe after every valid kg slot's last S2 MTE1 read
            wait round_{head_round-1}.W_free after every W slot's last S0 MTE1 read
            wait round_{head_round-1}.H_free after every H slot's last S0 MTE1 or S3 MTE3 access
            wait round_{head_round-1}.terminalDrain for all asynchronous transfers
            # 以上 round barrier 完成前，禁止发起本 round 的 kg/H/W 预取
        if initial_state is FP32:
            S-1 SinitCastFP32ToBF16（每个active head一次）
            drain all active S-1 input/H banks
        for chunk:
            tokenBegin = tokenBase(sequence, chunk)
            globalChunkId = dense ? chunk : chunkPrefix(sequence) + chunk
            hasP = !(chunk==0 && initial_state is None)
            if hasP:
                S0
            if chunk is final && !output_final_state:
                S1(v_new-only, with-P/no-P selected by hasP)
                continue
            S1(full, with-P/no-P selected by hasP) -> S2 -> S3
```

上式中所有 `k/w/u/g/gk/v_new` 地址都以 `tokenBegin` 为 token 轴基址，所有 `h` 地址都以
`globalChunkId` 为 chunk 轴下标。dense 模式的 batch 轴继续参与外层物理地址；varlen 模式
的物理 batch 固定为 0，不能用 sequence-local `chunk` 直接覆盖其他 sequence 的 `h` 槽。

不能使用 `chunk -> head_round`：当单核 head 数超过 4 时，后一个 head round 会在前一个
round 进入下一 chunk 前覆盖其 UB/L1 resident。

同一个 round 内，AIC 最多四个 head slot；AIV0 负责 round-local head 0/2，AIV1 负责
1/3。每个 head 的 Stage 实例完整执行，round 中不同 head 可以按 ready/free wavefront 推进。
P、D 分阶段占用的每-head local data slot和L1的每-head resident slot保证这种wavefront不依赖其他head
的执行先后避免覆盖。

head round 之间不允许直接重置 owner。`head_round=r` 的所有 chunk 都完成后，scheduler 必须
先等待该 round 的每个有效 `kg` slot 收到最后一个 S2 MTE1 消费者的
`kgOverwriteSafe`，等待每个 W slot 完成最后一次 S0 MTE1 读取并归还 `W_free`，等待每个 H
slot 完成最后一次 S0 MTE1 或 S3 MTE3 访问并归还 `H_free`，再等待所有异步搬运的
`terminalDrain`；只有这些事件全部闭合，才可以为 `head_round=r+1` 发起新的 `kg`、H 和 W
MTE2 预取。即使下一 round 使用相同的 `hk`，也必须在该 round barrier 之后重新从 GM 读取，
不能让下一轮预取覆盖上一轮尚未消费的 kg、H 或 W。

`head_round -> chunk` 不允许把 kg 带到下一 round。本文只为当前 chunk/round 的实际
`required_hk_round` 保留 kg slot，并在最后一个当前消费者完成后复用。因此：

```text
cache key = (sequence, chunk, kh)，仅在当前 round 有效
active_hv_round = [4*roundId, min(4*(roundId+1), HV))
g-only:  kh = hk = hv_to_hk(hv)，映射到同一 hk 的 value head共享一个 raw-k slot
gk-only: kh = hv，Prepare kg按value head独立，禁止跨hv共享
required_hk_round = unique({kh : hv in active_hv_round})
Nkg_round = |required_hk_round|，且 1 <= Nkg_round <= active_hv_count <= 4
每个当前 round/chunk：只按 required_hk_round 分配 Nkg_round 个 slot；同一 kh 只做一次 GM->L1，
               供当前 round 映射到它的 head 共享；最后一个当前消费者 MTE1 完成后失效
```

每次为 Stage2 准备 `kg`/`k_raw` 前，scheduler 必须先完成上述去重并得到
`required_hk_round`，然后为每个 required `kh` 建立或复用一个 slot；Stage2 只加载这些
slot 参与 MMAD。`Nkg_round` 是当前 round 的 distinct `(chunk,kh)` 数，而不是整个
sequence 的 key 总数。由于每个 round 最多 4 个 active head，`Nkg_round` 最多为 4；不把
下一 round 的 key 预取或留在 L1，因此同一 `hk` 跨 round 时必须重新从 GM 读取，不形成
跨 round 复用，也不需要淘汰模式或全局 key-count 分支。

### 5.6 同步协议

每个 head slot 使用成对的 ready/free 信号：

```text
首chunk BF16 initial:
  AIC本地 MTE2(initial GM->L1) -> MTE1，不等待跨核H ready

首chunk FP32 initial:
  所有active S-1 AIV H0->GM/L1 MTE3完成并phase drain
  -> 按active mask统一发布H ready -> 各head S0 MTE1

首chunk无 initial:
  无H ready、无S0、无P ready；S1从FREE resident slot开始写`V_new`

后续chunk:
  上一S3 AIV MTE3(Hnext->L1) -> H ready

通用有S0链:
  H ready（BF16首chunk为AIC本地ready）
  -> S0 MTE1 consume
  -> P ready (PIPE_FIX)
  -> S1 VF/MTE3
  -> g-only发布`V_new_g` ready；gk-only发布`V_new` ready in L1 (PIPE_MTE3)
  -> S2 MTE1 consume
  -> D ready (PIPE_FIX)
  -> S3 VF/MTE3
  -> H ready for next chunk (PIPE_MTE3)

local data slot 分 Stage ownership:
  S0 Fixpipe写P后发布P ready；S1 VF最后读取P后才结束P语义
  S1所有以该local data slot为源的MTE3完成后，才发布当前分支的Stage1右操作数ready
  首chunkno-initial且StateT=FP32时，这包括H0->h GM和`V_new_g`->L1两路MTE3
  S2必须等待该ready证明前一owner已free，才能用Fixpipe把同一local data slot写为D
  S3等待D ready，正向读取D并在同一次VF中把FP32分支的BF16 H写入本head固定低32KiB
  FP32分支必须等H->h GM/L1两路MTE3完成后才归还local data slot；BF16分支在D最后读取后即可归还
  full链中，g-only的`V_new_g` ready或gk-only的`V_new` ready同时证明上一P已free；
      H ready同时证明上一D owner已free
  v_new-only且本round存在P时，不产生`V_new_g` ready：S1 VF最后读取P后，若下一实际写者是AIC S0，
      以PIPE_V发布独立unionFree并由该S0 wait；若下一round无initial而跳过S0，则不向AIC set
      下一no-P VF与P最后读取同属本AIV的V pipe，按程序顺序执行；需要显式V内存栅栏时只用
      PIPE_V barrier，不生成跨pipe/跨核token
  最终S3不产生下一H ready：D最后读取完成，且FP32分支不存在H MTE3后，若下一实际写者是
      AIC S0，才以PIPE_V发布独立unionFree并由该S0 wait；下一round无initial时不向AIC set
  首chunkno-initial且v_new-only的FP32分支在union低32KiB生成H0并发起h0 GM MTE3:
      next=下一round no-P S1 VF时，set/wait unionMte3ToVFree，HardEvent=MTE3_V
      terminal时等待h0 MTE3完成后drain/release，不生产跨核unionFree
  无initial的full链由下一round S1在本AIV顺序执行；其`V_new_g` ready在AIC S2写D前建立跨核顺序，
      不需要额外发给不存在S0的unionFree

  unionFree仅在下一实际写者为AIC S0时，按物理head/local data slot配置为AIV->AIC跨核信号；
  1AIC:2AIV模式下owner AIV在实际
  最后读取后参与set，non-owner AIV在同一调度分支参与空set，AIC只在确有下一round写P时wait。
  三方对“是否存在下一round”的判断必须一致，禁止仅owner set或产生无消费者token。

kg slot:
  current-round slot FREE
  -> scheduler先计算required_hk_round；可在S0期间为每个required (chunk,kh)执行MTE2 clear entry + copy valid rows
     （无S0或未预取时在S2前只补齐缺失slot）
  -> first mapped head waits MTE2->MTE1 ready once; entry becomes valid
  -> current round 中映射到该 kh 的 head 依次 MTE1 consume；同一kh不重复GM->L1
  -> 最后一个当前 round 消费者返回 kg overwrite-safe
  -> slot 立即 invalid/free；下一 round 必须从 GM 重新装载

S-1 FP32 initial/H banks:
  每个local slot独立绑定initialInput[slot] 64 KiB和hOutput[slot] 32 KiB
  -> initial GM MTE2 -> MTE2_V ready -> 该slot唯一一次VF
  -> VF最后读取initialInput后归还该输入bank；VF写hOutput后发布V_MTE3 ready
  -> 同一hOutput依次发起h GM和L1 resident两路MTE3；全部完成后归还该H bank并只记录localDone
  -> 两个slot可以ping/pong，但每个slot不得在MTE3仍读取H时由VF覆写
  -> 当前head round的所有active input/H bank全部drain后，才统一发布各active head的H ready；
     AIC在此之前不得启动任一S0，AIV也不得进入main chunk loop
  -> 下一head round进入S-1前，上一round main loop也必须先drain与[0,192)重叠的实际owner
  上述drain是明确的局部event收尾，不是重置owner，也不使用SyncAll

V_new work banks:
  sequence/kernel首次进入，或S-1 phase完成上述drain后，bank才处于owner=FREE
  head round切换不得在未drain前重置owner；没有前序异步访问时首次S1不等待事件
  S1 U MTE2:
      owner=FREE时直接写；上一owner=MTE3且next=MTE2时wait uvToMte2Free[slot]
  -> S1 VF
  -> g-only/v_new-only把`V_new` GM MTE3计为bank最后使用
  -> gk-only把`V_new` GM与L1 `V_new`两路MTE3均计为bank最后使用
  -> next=MTE2: set/wait uvToMte2Free[slot]，HardEvent=MTE3_MTE2
  -> next=下一head round的S-1 phase: 等待真实MTE3完成并terminal drain，再移交整段地址
  -> S3不复用V_new work bank；FP32 H在D逐行末次读取后写本head local data slot，BF16 H直接从state输出
  -> terminal分支等待实际MTE3完成后drain/release，不生产无消费者event

BF16 rolling state slot[2] [128,192):
  BF16 initial: 首个S1 MTE2一次搬入完整R0；no-initial由同一次no-P VF写零R0/H0
  -> BF16 initial先set/wait MTE2_MTE3，no-initial先set/wait V_MTE3
  -> 首个S1以R0为h0 MTE3源；全部h0 MTE3发起后按实际next owner发布free
  -> 每次S3原位更新R前，若上一owner=MTE3_V则wait bf16StateToVFree[slot]
  -> S3 VF原位生成Rnext后set/wait V_MTE3；非末chunk以Rnext按state_gm_offset写h GM并写L1，
     末chunk按需按state_gm_offset写final_state
  -> 以上任一以state为源的MTE3发起后，统一按实际下一写者选择且只选择一种free:
       next=本round后续S3 VF或下一round zero-state VF:
           set/wait bf16StateToVFree[slot]，HardEvent=MTE3_V
       next=下一head round的BF16-initial S1 MTE2:
           set/wait bf16StateToMte2Free[slot]，HardEvent=MTE3_MTE2
       terminal: 等待真实MTE3完成后drain/release，不生产无消费者event
  -> 单chunkv_new-only也使用上述next-owner分支，禁止无S3时固定发布MTE3_V
  -> 不得在MTE3仍读state时由VF/MTE2覆写

FP32 shared state scratch [160,224):
  S-1不把initial放入该shared scratch；S-1使用独立initialInput[2]并在phase结束时drain
  sequence/kernel首次进入，或S-1 phase完成且释放所有重叠bank后，owner=FREE
  不执行S-1的head round必须继承上一round最后一次MTE3访问留下的真实owner，不能盲目重置
  普通S3 state MTE2:
      owner=FREE时直接写；上一owner=MTE3时wait stateToMte2Free
  首chunkno-initial的zero-state VF:
      owner=FREE时直接写；上一owner=MTE3时wait stateToVFree
  -> VF原位生成Rnext
  -> Rnext有后续S3消费者或需输出final_state:
       发起rolling/final-state MTE3
       next=zero-state VF: set/wait stateToVFree，HardEvent=MTE3_V
       next=state MTE2: set/wait stateToMte2Free，HardEvent=MTE3_MTE2
       terminal: 等待实际MTE3完成后drain/release，不生产无消费者event
  -> Rnext没有状态消费者:
       不发起state MTE3，不生成event，VF完成后将scratch直接归还为FREE
```

生产者复用 UB/L1 slot 前必须 wait free；消费者读取前必须 wait ready。不得无消费地连续 set
同一 flag，也不得用 `SyncAll` 替代上述局部依赖。核内 MTE2/VF/MTE3 和 MTE1/MMAD/Fixpipe
事件同样按每个 ping/pong slot 动态分配并闭环释放。`stateToVFree`与
`uvToMte2Free`/`stateToMte2Free` 分别是 `MTE3_V` 和 `MTE3_MTE2` 的动态 EventID；BF16 state
的MTE3->VF复用使用按slot分配的 `MTE3_V` `bf16StateToVFree`，MTE3->下一round MTE2
复用使用独立的 `MTE3_MTE2` `bf16StateToMte2Free`。这些事件不得复用
同一个逻辑token或同一个未释放代际。scheduler在发起最后一条MTE3前即可确定下一owner，
no-initial v_new-only的H0 union MTE3->下一round no-P VF使用按slot分配的`MTE3_V`
`unionMte3ToVFree`，不得误发AIV->AIC `unionFree`。
每代只set一个实际会被wait的事件。kg 的 key、valid、consumer_count 属于 AIC 本地
调度状态；MTE2->MTE1 ready event 每次填充只消费一次，映射到同一 kh 的 head 只能在当前
round 复用该 token。当前 round 最后一个 MTE1 读取完成后立即释放 slot；下一 round 不读取
旧 slot，必须重新从 GM 装载。

### 5.7 固定地址、碎片与 head 一致性

UB/L1 的空洞是容量设计的一部分，不进行运行时 compact 或 defragment：

```text
S0 UB当前owner:
  BF16 P slot 0/1 = [0,16) / [64,80)
  FP32 P slot 0/1 = [0,32) / [64,96)

S1 UB当前owner:
  P保留在S0所列地址，直到本S1 VF末次读取
  g-only V_new_g slot 0/1 = [32,48) / [96,112)
  gk-only V_new slot 0/1 = 固定输入输出bank中的两个16 KiB区

S2 UB当前owner:
  FP32 D slot 0/1 = [0,64) / [64,128)

S3进入时UB当前owner:
  FP32 D slot 0/1 = [0,64) / [64,128)
  FP32 state分支逐行末次读取D后，才把该行低32KiB地址写为H

UB BF16 state slot 0/1 = [128,160) / [160,192)
UB BF16 path H source  = [128,192) BF16 state resident，不分配独立scratch
UB S-1 H bank 0/1      = [0,32) / [32,64)，每份完整32 KiB BF16
UB S-1 initial bank 0/1= [64,128) / [128,192)，每份完整64 KiB FP32
                         # 仅S-1 phase使用；两套bank全部drain后再切换到main布局
UB FP32 state scratch  = [160,224)，当前head完整64 KiB；仅main S3使用
UB FP32 S3 H target    = 当前head D完成逐行末次读取后的固定低32 KiB：[0,32)或[64,96)
UB gate/alpha slot 0/1 = 固定的[224,225) / [225,226)子区

S0 L1 head slot当前owner = H_c；[128,160) / [160,192) / [192,224) / [224,256)
S1 L1 head slot当前owner = g-only V_new_g_c，或gk-only V_new_c；地址同上
S2 L1 head slot当前owner = S1保留的当前分支右操作数；地址同上
S3 L1 head slot当前owner = H_{c+1}；地址同上
S0可预取的kg slot 0/1/2/3 = [256,272) / [272,288) / [288,304) / [304,320)
                           # 每份16 KiB；S1继续保留，S2末次本轮MTE1后释放；不跨round
```

S1 的派生结果和 FP32 `initial_state` 派生的 BF16 H 写入各自独立固定目标区。S-1 的
`[0,192)` 与 main 布局存在地址复用，但两个 phase 之间显式 drain 全部真实 owner，任一
时刻不存在活数据重叠，也不通过 UB 搬移切换语义。FP32 S3 正向 VF 先读取相应 D 行，确认
该行生命周期结束后，才在固定低地址写 H，不移动任何活数据。U 输入变为 V_new、R 变为
Rnext 都是同 dtype、同元素地址的逐元素 owner 移交：源元素先进入寄存器且末次读取完成，
再写回原地址，不产生 UB DataCopy。

L1 head slot 的 owner 移交严格发生在 Stage 边界：S0 的 H 完成末次 MTE1 后，S1 才写当前
gate 分支的右操作数；S2 完成该右操作数的末次 MTE1 后，S3 才写 Hnext。`kg` 不在 slot
间迁移；若由 S0 预取，则原地址一直保留到 S2 最后一个本轮消费者。S1 的 ND 输出经 MTE3
直接生成后续 Cube 所需的 L1 zN，不是 UB/L1 内整理搬移。

碎片证明按最大连续需求检查：UB 的64 KiB FP32 D与state、32 KiB BF16 H、16 KiB P以及
g-only的16 KiB `V_new_g`均已有独立连续区；
L1 的32 KiB MMAD/head resident和16 KiB kg slot也各有固定连续区。任何阶段都不需要拼接
空洞获得连续空间，禁止 UB->UB、L1->L1 compact/memmove/defragment。`state_v_first` 的
布局差异只由 kernel 在 GM 读写边界处理，禁止在算子外增加 L2 transpose 或依赖转置后的
临时 state Tensor。

同一 dispatch 内的 dtype、gate mode、`output_final_state` 和 initial-state presence 都是全局
属性，不能按 head 选择存储策略：

| 语义 | 所有 active head 的统一存储/操作 |
| --- | --- |
| `P` | S0 全部写各自 local data slot，并保留到 S1 末次读取；不允许部分 head 写 GM workspace |
| `D` | 前一 owner 全部释放后，S2 全部写各自 local data slot，并保留到 S3 末次读取 |
| g-only `V_new_g` | 全部 head写各自固定 `V_new_g` 目标区，再以相同方式 MTE3 到 L1 |
| gk-only `V_new` | 全部 head在固定V_new work bank中完成从U owner到V_new owner的原位移交，再以相同方式MTE3到L1 |
| `kg` | g-only按`required_hk_round`缓存raw k；gk-only按`hv`缓存kg；同一gate mode不混用key语义；实际槽数为`Nkg_round` |
| BF16 `R` | 全部 head使用 BF16 UB resident；不允许单独 head退回 GM |
| FP32 `R` | 有后续状态消费者时，全部 head统一保存于rolling GM；无消费者时全部 head统一不写GM。存在输入/输出时才执行对应完整MTE2/MTE3 |
| g-only L1右操作数 | 全部 head在S1写各自固定L1 resident slot，owner为`V_new_g`，保留到S2末次MTE1 |
| gk-only L1右操作数 | 全部 head在S1写各自固定L1 resident slot，owner为`V_new`，保留到S2末次MTE1 |
| `H` | S2右操作数owner释放后，S3才写同一L1 resident地址并保留到下一S0末次MTE1 |
| `g/gk_last/alpha` | 全部 head使用对应的固定小 UB slot |
| `v_new/h/final_state` | 作为公开输出，全部 head按同一属性写 GM |

partial round 只减少 active task 数，不改变剩余 head 的路径。FP32 state scratch 是所有
head依次经历的瞬时VF工作区，不是某个head的跨Stage驻留选择；需要持久化时，状态对所有
head始终统一位于rolling GM，不需要持久化时则统一省略state MTE3。kg 不存在
sequence 级 cache mode；每个当前 round/chunk 的 Cube 操作数都先放入对应 L1 slot，
当前 round 结束后统一失效，下一 round 从 GM 重载。

## 6. 规则逐项核对

| 规则 | 设计结论 |
| --- | --- |
| 1 | S0/S2 只有 Cube；S-1/S1/S3 只有 Vector；S-1仅FP32 initial执行一次 |
| 2 | S0/S2 均不消费本 Stage 输出；S1/S3 只在单次 VF 内前向依赖寄存器结果 |
| 3 | L1 仅 Cube 数据，UB 仅 Vector 数据，预留分别为 320 KiB/最高地址226 KiB，不超过硬件容量 |
| 4 | P[2]按PType为32/64 KiB；D[2]在两条路径均为完整FP32 128 KiB；分别保留到S1/S3最后读取 |
| 5 | BF16 R[2]、alpha[2] 在 UB 保留；需持久保存的FP32 R按规则 14 fallback |
| 6 | g-only的`V_new_g`、gk-only的`V_new`以及后续H按Stage依次使用四个32 KiB L1 resident slot；前一owner free后才移交 |
| 7 | 当前无 Cube->Cube 中间结果；kg 仅保留当前 round 最多四份，不跨 round 保留 |
| 8 | 每个 Stage 的全部活跃 Tensor 均在对应空间图内，峰值已按 dtype 分支核算 |
| 9 | 四份 head resident 和四份物理 kg slot独立预留；每轮只启用 `Nkg_round` 份，语义只在最后消费者归还 free 后改变 |
| 10 | W/U/g在task内只读一次；BF16 initial分别供Cube/Vector各读一次；当前 round 每个 distinct `(chunk,kh)` 只搬一次并可共享 |
| 11 | 每-head local data slot和L1 resident slot均按Stage只标一个当前owner，且每次地址移交都有明确末次消费者 |
| 12 | 单 head Stage 实例完整搬运、一次 VF/MMAD；尾块不增加 pass |
| 13 | Cube scratch、L1 resident、kg物理分离；两个local data slot物理分离；S-1两套bank物理分离并在main前完整drain |
| 14 | FP32 rolling state及首个S3对FP32 initial的再次读取、当前 round kg 重载，均有容量证明 |
| 15 | BF16 initial主链严格C/V/C/V；FP32 initial只增加一次必要的S-1 Vector；无initial跳过无效S0 |
| 16 | g-only alpha 在 S1 计算后保留到 S3；zero-state/final v-only 分支不做无效 gate 运算 |
| 17 | 活数据不搬移；派生输出写独立固定区或已完成最后读取的固定子区，连续空间已闭合，不做copy/compact/defragment |
| 18 | 存储策略按 dispatch统一；同语义 active head全部走 UB、L1或GM中的同一路径 |
| 19 | 四个主Stage是依赖下限；额外S-1仅由FP32 initial到BF16 Cube输入的硬件类型边界触发 |

## 7. 实现准入条件

在把本文方案实现为 kernel 前，必须满足：

1. Host 对 A5 目标路径显式拦截 `BT=64, K=128, V=128`，并且只接受BF16 `k/w/u`；
   报错、README 和 API 文档一致。
   `g`/`gk` 按 presence 恰好选择一个 gate mode；双空和双非空都返回参数错误。g-only 校验
   `HV % HK == 0`；gk-only 校验物理 `k=kg` 且 `k.shape[1] == gk.shape[1] == HV`，cache key
   分别使用 `hk`/`hv`，不得跨value head共享kg。StateT 严格按 `initial_state.dtype ->
   final_state.dtype -> FP32` 的优先级推导；先校验output属性与final指针presence一致，任一
   指针为空时不得解引用。varlen下还必须校验`cu_seqlens`严格递增，禁止空sequence。
2. scheduler 改为 `sequence -> head_round -> chunk`，并为四个 L1 resident slot建立 ready/free；
   每个当前 round/chunk 先计算 `required_hk_round`，只为 `Nkg_round` 个 distinct `(chunk,kh)` 建立
   kg ready/free；未使用的物理槽保持 FREE，round 结束后有效 slot 全部失效。
3. S0 的FP32 L0C结果按StateT选择PType：BF16路径使用`F322BF16`，FP32路径使用`NoQuant`；
   S2 的FP32 L0C结果在两条路径都使用`NoQuant`。S0的P和S2的D分别在各自Stage直接进入
   配对AIV UB，不再使用GM workspace。
4. S1 只调用一次完整 RegBase VF；g-only输出UB ND `V_new_g`，gk-only输出UB ND `V_new`，
   再由MTE3直接写L1 zN。
5. S3 只调用一次完整 RegBase VF；gate/state/output 分支在 VF 外选择具体 VF。
6. BF16 state 使用两份 UB resident；存在后继状态消费者时，FP32 state使用明确的 rolling GM
   workspace；单 chunk且不输出final_state时不物化无消费者的rolling state。
7. BF16 initial不创建S-1，首个S0直接读取initial；FP32 initial恰好执行一次完整S-1；
   无initial时彻底跳过首个S0，S1 no-P VF生成零H0，首个S3使用zero-state VF且不读取
   未初始化rolling GM。
8. 所有核内 EventID 动态分配、成对 wait/set/release；所有核间 flag 有 ready/free 收支证明。
9. 不在 kernel 中添加 shape/dtype/gate 模式 `static_assert`；非法输入只在 Host 侧拦截。
10. 实现后分别覆盖BF16数据面、BF16/FP32 StateT、BF16/FP32 GateT、g/gk、exp/exp2、有/无 initial state、
    output_final_state true/false、尾 chunk、varlen、1/2/3/4 active head、按`required_hk_round`得到的
    `Nkg_round=1/2/3/4`、同一`hk`跨round重复读取、
    GVA的共享raw-k和逐value-head kg，以及每核多 head round。公开入口和物理 Host 都要覆盖
    `state_v_first=false/true` 的原生 GM 输入/输出布局；两种布局均不得依赖外部 L2 transpose。
    对同一 logical state，`true` 与 `false` 的 `h/final_state` 结果必须满足末两轴转置等价；
    反向用例覆盖 state 末两轴与 `state_v_first` 不匹配、`g=None,gk=None`、`g!=None,gk!=None`
    等参数错误，不再把 `state_v_first=true` 作为不支持分支。
    StateT 推导不按无效笛卡尔积测试，而是逐项覆盖合法组合：initial_state为BF16/FP32且
    output=false/final=null；initial_state为BF16/FP32且output=true/final同dtype；initial为空且
    output=true/final为BF16或FP32；initial与final均为空时output=false并固定选择FP32。反向用例
    还要覆盖initial/final均非空但dtype不一致，以及output属性与final指针presence不一致；
    运行用例分别覆盖单chunk不物化rolling state和多chunk FP32 hidden rolling路径；反向覆盖
    FP16/FP32 `k/w/u`、空sequence拦截。还要静态/trace确认BF16 initial不进入S-1、FP32 initial只进入一次，
    并以对应golden验证两条分支进入S0/S2 MMAD的operand均为BF16、accumulator均为FP32；
    FP32 state分支在每个chunk边界先生成BF16 Cube shadow，且P和D均保持FP32；BF16 state分支
    P为BF16但D保持FP32，S3末尾才量化Rnext。
11. UB/L1 地址在编译期或 Init阶段固定；实现中不得加入 UB->UB/L1->L1 compact、memmove或
    为获得连续空间而进行的 tensor relocation。P与g-only的`V_new_g`、FP32 initial_state与
    BF16 H使用独立固定区；FP32 D逐行末次读取后才允许在同一head local data slot的固定低地址
    写H，并证明正向VF只覆盖已经最后读取的D地址。
12. 对同一 dispatch 的所有 active head做存储路径断言/静态审查：BF16 state全部驻 UB，
    需要持久保存的FP32 state全部走 rolling GM，S0的P与S2的D分别驻UB；S1当前分支右操作数
    和S3的H按owner移交驻L1；partial round
    不得改变该策略。
13. V_new work bank必须按5.6节的owner状态机逐代闭环：初始FREE不伪造wait；S1完成所有以该
    bank为源的MTE3后，下一owner为MTE2时只发布`MTE3_MTE2`的`uvToMte2Free`；下一round
    进入S-1或terminal时等待真实MTE3完成并drain。不能复用上一代token或依赖跨head自然顺序。
14. FP32 `[160,224)` state scratch只属于main S3，使用独立owner状态机；zero-state VF、
    rolling-state MTE2和state MTE3纳入同一代际。MTE3来源按下一owner只发布
    `stateToVFree`或`stateToMte2Free`；复用前显式wait，退出前按真实终态drain/release。
15. BF16 state slot同样按下一写者分型：本round S3或下一round zero-state VF使用
    `bf16StateToVFree`，下一round BF16-initial MTE2使用`bf16StateToMte2Free`；v_new-only、
    final-state和terminal路径不得固定发布无消费者event。
16. local data slot的末路径按实际下一写者分型：只有下一AIC S0存在时发布跨核`unionFree`；
    无initial的下一round不得向被跳过的S0发token。no-initial v_new-only的H0 MTE3必须以
    本核`unionMte3ToVFree`保护下一round no-P VF，最后round只drain不set。
17. S-1必须物理分配两份64 KiB FP32 initial输入bank和两份32 KiB BF16 H输出bank；每个
    local head只访问自己的bank。所有active initial bank被VF最后读取、两个H bank的h GM/L1
    MTE3全部完成且S-1 event完成drain后，才按active mask统一发布H ready并把`[0,192)`按main
    布局复用；AIC不得按单head ready提前写P。禁止退化为单H/shared-state scratch串行方案。
