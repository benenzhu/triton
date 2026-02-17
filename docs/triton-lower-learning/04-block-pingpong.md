# Triton Lowering 学习笔记 4: BlockPingpong

本文解析 AMD 后端的 `TritonAMDGPUBlockPingpong` pass，它通过在同一 SIMD 上交错调度两个 warp 的计算（Dot）和访存（Memory）操作，提升 MFMA 指令的利用率。

## 源码位置

| 文件 | 说明 |
|------|------|
| `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp` | pass 实现（~1274行） |
| `third_party/amd/backend/compiler.py:22-24,238-256` | 启用逻辑 |
| `test/TritonGPU/amd/amd-block-pingpong.mlir` | One/Two/Four PP Clusters 测试 |
| `test/TritonGPU/amd/amd-block-pingpong-chained-dots.mlir` | ChainedDot 测试 |

```python
# compiler.py 中的调用位置
use_async_copy = is_async_copy_enabled(options.arch)
use_block_pingpong = is_pingpong_schedule_enabled(options.arch, use_async_copy)

amd.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)
amd.passes.ttgpuir.add_pipeline(pm, use_async_copy, use_block_pingpong)
# ... 中间若干 pass ...
amd.passes.ttgpuir.add_move_up_prologue_loads(pm)
if use_block_pingpong and options.num_stages > 1:
    amd.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)
```

## 背景：CDNA SIMD 架构与 Warp 调度

### 为什么 Pingpong 有效

AMD CDNA 架构（gfx942/gfx950）中，每个 SIMD（Compute Unit 内的执行单元）可以同时容纳**多个 warp**（wavefront）。关键硬件特性：

1. **MFMA 指令延迟高**：一条 `v_mfma_f32_32x32x16_bf16` 需要数十个周期完成
2. **SIMD 可交替执行不同 warp**：当 warp A 在等待 MFMA 结果时，SIMD 可以切换到 warp B 执行访存指令
3. **计算与访存可重叠**：MFMA 使用矩阵核心，load/store 使用内存通道，硬件上可以并行

Pingpong 调度正是利用这一特性：将循环体分为 **Dot cluster**（MFMA 计算）和 **Memory cluster**（global/local load/store），让同一 SIMD 上的两个 warp 交替执行不同类型的 cluster，实现计算与访存的重叠：

```
SIMD 上的两个 warp 交替执行:

Warp A: [===Dot===][===Memory===][===Dot===][===Memory===]
Warp B: [===Memory===][===Dot===][===Memory===][===Dot===]
              ↑                         ↑
         Dot 和 Memory 在硬件上重叠执行
```

### 前提条件

- 必须是**计算密集型**（compute-bound）的 kernel。如果是访存密集型，重叠计算没有意义
- 需要 **Pipeline pass 已经执行**（`num_stages > 1`），用软件流水线隐藏 global memory 延迟
- Pingpong 解决的是 **MFMA 与 LDS 访问之间的延迟**，而非 global memory 延迟

## 触发条件

### 编译器启用逻辑

```python
# compiler.py
def is_pingpong_schedule_enabled(arch, use_async_copy):
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)) \
        if knobs.amd.use_block_pingpong is None else knobs.amd.use_block_pingpong
```

- **gfx942**（MI300X）：默认启用
- **gfx950**（MI355X）：仅当 `use_async_copy=True` 时启用（gfx950 默认开启 async copy）
- 可通过 `knobs.amd.use_block_pingpong` 手动覆盖

### getDotPingponged() 中的检查

pass 入口 `getDotPingponged()` 对循环体执行一系列检查：

**1. num_stages 检查**（line 948）
```cpp
if (numStages <= 1) return;  // 需要 pipeline 已执行
```

**2. 收集循环内的 op**（line 959-987）

遍历 `scf.for` 的循环体，按类型收集：
- `tt.LoadOp` → `gLoadOps`（global load）
- `ttg.LocalLoadOp` → `lLoadOps`（LDS → register，只收集软件流水线的）
- `ttg.LocalStoreOp` → `lStoreOps`（register → LDS）
- `tt.DotOp` → `dotOps`
- `tt.DotScaledOp` → `scaledDotOps`（FP8 scaled dot）
- `ttg.AsyncCopyGlobalToLocalOp` → `asyncCopyOps`

注意 `lLoadOps` 只收集**来自循环携带变量**（loop carried values）的 local_load，即经过 pipeline 展开后的操作。

**3. Dot-like op 数量检查**（line 997）
```cpp
int64_t numOfDotLikeOps = scaledDotOps.size() + dotOps.size();
if (numOfDotLikeOps < 1 || numOfDotLikeOps > 2) return;  // 只处理 1 或 2 个 dot
```

**4. 2 个 dot 的情况**（line 1002）：走 ChainedDot 路径，要求 `numStages == 4`

**5. 1 个 dot 的情况**：需要 `gloadSize >= 2`，`lLoadOps >= 2`，然后根据 tile 大小、numWarps、numStages 选择具体模式

## 调度模式总览

pass 根据不同的配置选择不同的调度策略：

| 模式 | 条件 | 适用场景 |
|------|------|----------|
| **One PP Cluster** | `numWarps=4`, tileSize ∈ [minTile, smallTile] | 小 tile（如 128×128×64 fp16），跨 block 的 warp 交错 |
| **Four PP Clusters** | `numWarps=8`, `numStages=2`, tileSize ≥ largeTile | 大 tile（如 256×256×64 fp16），dot 切成 4 份 |
| **Two PP Clusters** | `numWarps=8`, `numStages=2`, tileSize == mediumTile | 中 tile（如 256×128×64 fp16），dot 切成 2 份 |
| **ChainedDot** | 2 个 dot ops, `numStages=4` | FAv3 等链式 dot 场景 |
| **TwoClusterWithLocalLoadAndAll** | `numWarps=8`, `numStages=3`, async copy, 1 个 dot | 大 LDS + async copy 场景 |
| **TwoClusterWithAsyncAndAll** | `numWarps=8`, `numStages=2`, 1 个 scaledDot, async copy, 256×256 fp8 | FP8 scaled dot 场景 |

### Tile 大小定义

```cpp
const int64_t minTile = 262144;      // 32×128×64×16bit
const int64_t smallTile = 16777216;  // 128×128×64×16bit
const int64_t mediumTile = 33554432; // smallTile × 2（如 256×128×64×16bit）
const int64_t largeTile = 67108864;  // 256×256×64×16bit
```

tileSize 计算公式：`dotShape[0] × dotShape[1] × aShape[1] × elemWidth`

## 详解各模式

### One PP Cluster（numWarps=4，小 tile）

最简单的模式。不切分 dot，只重排操作顺序并插入 `s_setprio` 指令。适用于 `numWarps=4` 的场景——每个 SIMD 运行来自**不同 block** 的两个 warp，它们之间不需要 barrier 同步。

#### 变换逻辑

```
变换前（循环体内的操作顺序）：
  local_load A
  local_load B
  global_load A (next)
  global_load B (next)
  dot A×B
  local_store A
  local_store B

变换后：
  Memory Cluster:
    sched_barrier(1)        // 阻止访存指令跨越边界
    s_setprio 1             // 提高当前 warp 优先级
    global_load A           // global load 与 local load 交错排列
    sched_barrier(0)
    local_load B
    s_setprio 0             // 降低优先级
    global_load B

  Dot Cluster:
    s_setprio 1             // dot 执行时提高优先级
    dot A×B
    s_setprio 0             // dot 完成后降低优先级
```

- `sched_barrier(1)` 的 mask=1 仅阻止访存指令跨越，允许其他指令调度
- `sched_barrier(0)` 的 mask=0 阻止**所有**指令跨越

#### 测试用例（from `amd-block-pingpong.mlir`）

```mlir
// CHECK-LABEL: pingpong_small
// CHECK: ttg.local_load
// CHECK: rocdl.s.setprio 1
// CHECK: tt.load           ← global load A
// CHECK: rocdl.sched.barrier
// CHECK: ttg.local_load
// CHECK: rocdl.s.setprio 0
// CHECK: tt.load           ← global load B
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot            ← dot 被 setprio 1/0 包围
// CHECK: rocdl.s.setprio 0
```

**注意**：`numWarps=4` 不需要 `addAsymmetricSyncToLoop`，因为两个 warp 来自不同 block，天然独立。

### Four PP Clusters（numWarps=8，大 tile）

当 tile 很大（如 256×256×64 fp16）时，一个 dot 需要的寄存器量太大，无法在一个 cluster 里放完所有 local_load 数据再执行整个 dot。解决方案：**将 dot 切成 4 份**，每份配一对 mem/dot cluster。

#### Dot 切分（sliceDot）

`sliceDot()` 将一个 dot 沿 K 维度切成 `numSlices` 份：

```
原始:  dot(A[256×64], B[64×256]) → C[256×256]

切成 4 份:
  dot0: dot(A[256×16], B[16×256]) → C       // K 维度 [0:16]
  dot1: dot(A[256×16], B[16×256]) → C       // K 维度 [16:32]
  dot2: dot(A[256×16], B[16×256]) → C       // K 维度 [32:48]
  dot3: dot(A[256×16], B[16×256]) → C       // K 维度 [48:64]
```

每个 slice 通过 `ttg.MemDescSubsliceOp` 从 shared memory 的不同偏移读取。

#### 变换后的循环体

```
mem0: global_load A, local_load A(1/4), local_load B(1/4)
      [barrier + sched_barrier]
dot0: s_setprio 1 → dot A(1/4) × B(1/4) → s_setprio 0
      [barrier + sched_barrier]
mem1: global_load B, local_load A(2/4), local_load B(2/4)
      [barrier + sched_barrier]
dot1: s_setprio 1 → dot A(2/4) × B(2/4) → s_setprio 0
      [barrier + sched_barrier]
mem2: local_load A(3/4), local_load B(3/4), local_load A(4/4), local_load B(4/4)
      [barrier + sched_barrier]
dot2: s_setprio 1 → dot A(3/4) × B(3/4) → s_setprio 0
      [barrier + sched_barrier]
mem3: local_store A, local_store B
      [barrier + sched_barrier]
dot3: s_setprio 1 → dot A(4/4) × B(4/4) → s_setprio 0
      [barrier + sched_barrier]   ← 循环末尾
```

每对 cluster 之间插入 `ttg.barrier local`（用于非对称同步）+ `rocdl.sched.barrier 0`（阻止后端重排）。

### Two PP Clusters（numWarps=8，中 tile）

中等大小的 tile（如 256×128×64 fp16）。dot 切成 2 份。

#### 变换后的循环体

```
Memory Cluster #0:
  local_load A(1/2), local_load B(1/2)
  sched_barrier(0)
  global_load A
  sched_barrier(0)
  local_load A(2/2), local_load B(2/2)
  sched_barrier(0)
  global_load B
  s_barrier                    ← 硬件 barrier（不走 ttg.barrier）
  sched_barrier(0)

Dot Cluster #0:
  s_setprio 1 → dot(1/2) → s_setprio 0
  [barrier + sched_barrier]

Memory Cluster #1:
  local_store A, local_store B
  [barrier + sched_barrier]

Dot Cluster #1:
  s_setprio 1 → dot(2/2) → s_setprio 0
  [barrier + sched_barrier]   ← 循环末尾
```

与 Four PP Clusters 的区别：
- Memory Cluster #0 中 local_load 和 global_load 交错排列，用 `sched_barrier(0)` 维持顺序
- 第一个 cluster 使用 `rocdl.s.barrier`（直接的 `s_barrier` 指令）而非 `ttg.barrier local`

### ChainedDot Schedule（2 个 dot ops, numStages=4）

专为 **Flash Attention v3**（FAv3）等含两个连续 dot 的场景设计。Pipeline 已经以 `num_stages=4` 展开，循环体天然包含 2 对 compute/memory cluster。Pingpong 只需插入 barrier 和优先级控制。

#### 核心设计

与其他模式不同，ChainedDot 给**Memory Cluster 更高优先级**。

原因：compute cluster 以 MFMA 为主（占用矩阵核心），memory cluster 包含少量 `v_xxx`（VALU）指令用于地址计算。如果 compute cluster 优先级更高，它会垄断 VALU 发射槽，阻塞 memory cluster 的地址计算指令，破坏重叠。Memory cluster 优先级更高时，VALU 地址指令能及时发射，而 MFMA 不受影响（使用独立硬件单元）。

#### 变换后的循环体

```
Compute Cluster 1:
  s_barrier
  sched_barrier(0)
  dot0                         ← 第一个 dot

Memory Cluster 1:
  sched_barrier(0)
  s_setprio 1                  ← memory cluster 高优先级
  async_wait / barrier
  local_load
  async_copy + commit          ← 预取下一批数据
  sched_barrier(0)

Compute Cluster 2:
  s_setprio 0                  ← 降回低优先级
  s_waitcnt lgkmcnt(0)         ← 等待 LDS 读完成
  s_barrier
  sched_barrier(0)
  dot1                         ← 第二个 dot

Memory Cluster 2:
  sched_barrier(0)
  s_setprio 1
  async_wait / barrier
  local_load
  async_copy + commit
  sched_barrier(0)

Loop End:
  s_setprio 0
  s_waitcnt lgkmcnt(0)         ← 确保 ds_read 完成
  s_cbranch                    ← 循环跳转
  s_barrier                    ← 放在循环开头而非结尾
```

关键设计决策（来自源码注释）：

1. **s_xxx 指令放在 Memory Cluster**：`s_xxx` 和 `v_xxx` 只有来自不同 warp 时才能 co-issue。Compute cluster 以 VALU 为主，所以 `s_xxx` 放在 memory cluster 能最大化 co-issue 机会

2. **`s_waitcnt lgkmcnt(0)` 放在 Memory Cluster 末尾**：确保 `ds_read` 完成后再进入 compute cluster，避免 LLVM 后端在 compute cluster 内部插入额外的 `s_waitcnt`

3. **`s_barrier` 放在循环开头**：如果放在末尾，LLVM 后端可能把循环归纳变量的 `s_xxx` 指令调度到 `s_barrier` 之后，导致它们落入 compute cluster

### TwoClusterWithLocalLoadAndAll（async copy, numStages=3）

适用于大 LDS、async copy 场景。不切分 dot，而是用半大小的 `tile_K` 配合 `num_stages=3` 预取数据。

#### 变换后的循环体

```
Memory Cluster:
  local_load A (LDS → register)
  local_load B
  sched_barrier(0)
  async_copy A                 ← 发起下一批的异步拷贝
  async_commit
  sched_barrier(0)
  async_wait                   ← 等待前面的 async copy
  sched_barrier(0)
  sched_group_barrier(MFMA, 1) ← 提示后端交错调度
  sched_group_barrier(SALU, 3)
  sched_group_barrier(MFMA, 1)
  sched_group_barrier(SALU, 3)
  sched_group_barrier(MFMA, 1)
  async_copy B
  async_commit
  dot A×B                      ← 计算
  sched_barrier(0)
  s_barrier
  sched_barrier(0)
```

`sched_group_barrier` 提示后端每 1 条 MFMA 之间插入 3 条 SALU 指令，改善指令级并行。

### TwoClusterWithAsyncAndAll（scaled dot, async copy）

专为 FP8 `DotScaledOp` 设计。条件：`numWarps=8, numStages=2, 256×256 fp8, async copy`。

```
Memory Cluster:
  async_copy + commit (所有)
  global_load (如果有 LDS bypass)
  sched_barrier(0)
  s_barrier
  sched_barrier(0)

Compute Cluster:
  scaledDot {pingpong_2step}   ← 标记 pingpong_2step 属性
```

`pingpong_2step` 属性触发 TTGIR → LLVM lowering 阶段的第二步变换：将 dot 内部的 MFMA 分成两组，让 `ds_read` 只与第一组 MFMA 共存。

## 非对称同步（Asymmetric Sync）

除了 `numWarps=4` 的 One PP Cluster 模式外，所有模式都需要 `addAsymmetricSyncToLoop()`。

### 问题

`numWarps=8` 时，一个 block 有 8 个 warp。Pingpong 需要让半数 warp 在循环的不同位置执行：当 warp 0-3 执行 Dot cluster 时，warp 4-7 应该执行 Memory cluster。但循环开始时所有 warp 在同一位置。

### 解决方案：`amdg.cond_barrier`

```cpp
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
    // 1. 循环前插入全局 barrier
    triton::gpu::BarrierOp::create(builder, loc, ...);

    // 2. 计算 warp 分组
    workIDX = rocdl.workitem.id.x           // 线程 ID
    warpIDX = workIDX / 256                 // warp ID (warp_size=64, 4 warps per group)
    warpLow  = (warpIDX == 0)              // 前半 warp
    warpHigh = (warpIDX != 0)              // 后半 warp

    // 3. 循环前：阻塞后半 warp
    amdg.cond_barrier warpHigh             // warpHigh=true 的 warp 等待

    // 4. 循环后：阻塞前半 warp（让后半追上来）
    amdg.cond_barrier warpLow              // warpLow=true 的 warp 等待
}
```

效果：

```
循环前:
  [全局 barrier]           ← 所有 warp 同步
  [cond_barrier warpHigh]  ← 后半 warp 等待

循环中:
  迭代 i:
    warp 前半: [===Dot===][===barrier===][===Memory===][===barrier===]
    warp 后半:            [===Memory===][===barrier===][===Dot===][===barrier===]
                           ↑ 后半 warp 晚半拍进入循环

循环后:
  [cond_barrier warpLow]   ← 前半 warp 等待后半追上
```

循环内部的 `ttg.barrier local` + `rocdl.sched.barrier 0` 在每个 cluster 边界同步，但由于 `cond_barrier` 的错位，两组 warp 到达 barrier 时处于不同的 cluster，实现了交错。

## 关键硬件指令

| 指令 | IR 表示 | 作用 |
|------|---------|------|
| `s_setprio N` | `rocdl.s.setprio N` | 设置当前 warp 在 SIMD 内的优先级（0=低，1=高）。当两个 warp 竞争同一类型指令的发射槽时，高优先级的 warp 优先 |
| `s_barrier` | `rocdl.s.barrier` | 硬件 barrier，block 内所有 warp 必须到达后才能继续 |
| `s_sched_barrier mask` | `rocdl.sched.barrier mask` | 指令调度屏障。告诉编译器后端不要跨越此点重排指令。`mask=0` 阻止所有类型，`mask=1` 仅阻止访存 |
| `s_sched_group_barrier type,count,id` | `rocdl.sched.group.barrier type,count,id` | 提示后端按指定模式交错调度不同类型的指令 |
| `s_waitcnt lgkmcnt(N)` | `amdg.memory_counter_wait ds(N)` | 等待 LDS 相关操作完成 |
| (conditional barrier) | `amdg.cond_barrier %pred` | 条件 barrier：仅 `%pred=true` 的 warp 参与等待 |
| (block barrier) | `ttg.barrier local` | Triton 的 barrier（由 membar analysis 识别，避免插入重复 barrier） |

### s_setprio 的作用

Pingpong 的核心机制。两个 warp 在同一 SIMD 上执行，它们可能同时有就绪的指令：

- **无 setprio 时**：硬件随机或轮转选择 warp 发射——可能一个 warp 的 dot 还没执行完，另一个 warp 的 dot 就开始了，破坏 pingpong 模式
- **有 setprio 时**：执行 dot 的 warp 设 `setprio 1`，确保它独占 MFMA 发射直到完成，然后 `setprio 0` 让出给另一个 warp

## 在我们的 GEMM 上未生效的原因

根据 CLAUDE.md 的记录，我们的 MI355X GEMM kernel 配置为：

```python
BLOCK_M=256, BLOCK_N=256, BLOCK_K=64
num_warps=8, num_stages=2
```

Pipeline 展开后（`num_stages=2`），循环体包含 **2 个 `tt.dot` op**——一个来自当前迭代的 stage 1，另一个是下一迭代的 stage 1 被展开到本迭代中。

查看 `getDotPingponged()` 的逻辑：

```cpp
int64_t numOfDotLikeOps = scaledDotOps.size() + dotOps.size();

if (numOfDotLikeOps == 2) {
    if (numStages != 4)    // ← 我们是 numStages=2，这里直接 return！
        return;
    // ChainedDot 路径...
}
```

2 个 dot ops 的情况只有 `numStages == 4` 才走 ChainedDot 路径。而 `numStages=2` 时直接被跳过。

继续往下，`dotSize != 1` 的检查也会拦截：

```cpp
if ((gloadSize < 2 || lLoadOps.size() < 2 || dotSize != 1)) {
    // "Unable to match ping pong scheduling pattern"
    return;
}
```

所以 pass 在收集完 op 后就退出了，不会生成任何 `s_setprio` 或 `sched_barrier`。这可以通过检查编译产物的汇编确认——不包含这些指令。

### 可能的解决方向

- 使用 `num_stages=1` 避免 pipeline 展开产生 2 个 dot（但会失去流水线收益）
- 使用 `num_stages=4` 配合 ChainedDot 模式（但 LDS 可能不够：需要 4 份 buffer）
- 等上游扩展 pingpong 支持 `num_stages=2` + 2 个 dot 的场景

## 新增 Op 总结

| Op | 说明 | 来源 |
|----|------|------|
| `rocdl.s.setprio N` | 设置 warp 优先级 | ROCDL dialect |
| `rocdl.sched.barrier mask` | 指令调度屏障 | ROCDL dialect |
| `rocdl.sched.group.barrier type,count,id` | 分组调度屏障 | ROCDL dialect |
| `rocdl.s.barrier` | 硬件 barrier | ROCDL dialect |
| `amdg.cond_barrier %pred` | 条件 barrier（部分 warp 同步） | TritonAMDGPU dialect |
| `amdg.memory_counter_wait ds(N)` | 等待内存计数器 | TritonAMDGPU dialect |
| `ttg.MemDescSubsliceOp` | 从 memdesc 切片（dot slicing 用） | TritonGPU dialect |
