# Triton Lowering 学习笔记 2: ScheduleLoops & Pipeline

本文解析两个关键的 AMD 后端 pass：
1. **TritonAMDGPUScheduleLoops** — 给循环中的 op 标注调度阶段（stage）和集群（cluster）
2. **TritonAMDGPUPipeline** — 根据标注执行软件流水线变换，实现 load 与 compute 的重叠

## 源码位置

| Pass | 源码 |
|------|------|
| ScheduleLoops | `third_party/amd/lib/TritonAMDGPUTransforms/ScheduleLoops.cpp` |
| Pipeline | `lib/Dialect/TritonGPU/Transforms/Pipeliner/` |
| 注册位置 | `third_party/amd/backend/compiler.py:240-241` |

```python
# compiler.py 中的调用顺序
amd.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)
amd.passes.ttgpuir.add_pipeline(pm, use_async_copy, use_block_pingpong)
```

## Pass 1: ScheduleLoops — 标注调度信息

### 输入（ScheduleLoops 之前）

循环体是一个简单的 `scf.for`，每次迭代：load → convert_layout → dot → addptr。

```mlir
%acc:3 = scf.for %i = %c0 to %N step %c1
    iter_args(%acc_val, %a_ptrs, %b_ptrs) {

  // 所有操作在同一个 "stage"，没有重叠
  %a = tt.load %a_ptrs           // global load A tile
  %b = tt.load %b_ptrs           // global load B tile
  %a_cvt = ttg.convert_layout %a // 转为 dot operand layout
  %b_cvt = ttg.convert_layout %b
  %acc_new = tt.dot %a_cvt, %b_cvt, %acc_val  // MFMA 计算
  %a_next = tt.addptr %a_ptrs, %cst_64        // 指针步进
  %b_next = tt.addptr %b_ptrs, %cst_64
  scf.yield %acc_new, %a_next, %b_next
}
```

### 输出（ScheduleLoops 之后）

ScheduleLoops **不改变 IR 结构**，只给 op 添加属性标注：

```mlir
%a = tt.load %a_ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32}
%b = tt.load %b_ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32}
%acc_new = tt.dot %a_cvt, %b_cvt, %acc_val {loop.cluster = 1 : i32, loop.stage = 1 : i32}
```

并在循环上标注 `tt.scheduled_max_stage`：

```mlir
} {tt.scheduled_max_stage = 1 : i32}
```

### 标注含义

- **`loop.stage`**: 该 op 属于流水线的第几个阶段。`stage=0` 表示 prologue 阶段（load 数据），`stage=1` 表示 compute 阶段（执行 dot）
- **`loop.cluster`**: 将同一 stage 内的 op 进一步分组为集群。同一 cluster 的 op 在最终调度中尽量靠近
- **`tt.scheduled_max_stage`**: 循环的最大 stage 数 = `num_stages - 1`。`num_stages=2` 时 max_stage=1

### 为什么需要这一步

Pipeline pass 本身是通用的（在 `lib/` 下而非 `third_party/amd/`），它不知道哪些 op 该提前执行。ScheduleLoops 作为 AMD 特定 pass，根据 CDNA 架构的延迟特征决定：
- memory ops（`tt.load`）延迟高 → 放到 stage 0，需要提前发射
- compute ops（`tt.dot`）→ 放到 stage 1，消费前一次迭代 load 的数据

## Pass 2: Pipeline — 软件流水线变换

Pipeline pass 读取 ScheduleLoops 的标注，执行经典的软件流水线展开（software pipelining）。

### 核心变换

对于 `num_stages=2` 的循环，pipeline 将其变换为三部分：

```
Prologue:  执行第 0 次迭代的 stage 0 (load 第一批数据)
Main Loop: 第 i 次迭代同时执行 stage 0 (load 第 i+1 批) + stage 1 (compute 第 i 批)
Epilogue:  执行最后一次迭代的 stage 1 (compute 最后一批)
```

### 输出 IR 详细解析

#### 1. Shared Memory 分配（双缓冲）

```mlir
// 分配 2 份 shared memory（num_stages=2），交替使用
%a = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xbf16, #shared, #smem, mutable>
%b = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xbf16, #shared1, #smem, mutable>
```

`memdesc<2x256x64>` 中的 `2` 就是 `num_stages`。两个 buffer 交替使用：当 stage 0 往 buffer[1] 写入下一批数据时，stage 1 从 buffer[0] 读取当前数据。

#### 2. Prologue（循环前：预加载第一批数据）

```mlir
// 选择 buffer index: 第一次用 buffer[0]
%acc_40 = arith.select %acc_39, %c0_i32, %c0_i32 : i32
%a_41 = ttg.memdesc_index %a[%acc_40]     // 取 buffer[0]

// Async copy: global memory → shared memory（DMA，不阻塞）
%a_43 = ttg.async_copy_global_to_local %a_ptrs_28, %a_41 mask %acc_42
%a_44 = ttg.async_commit_group tokens %a_43  // 提交 async group

// 同样对 B 做 async copy
%b_49 = ttg.async_copy_global_to_local %b_ptrs_37, %b_47 mask %acc_48
%b_50 = ttg.async_commit_group tokens %b_49
```

关键 op：
- **`ttg.async_copy_global_to_local`**: AMD 的 async global → LDS 拷贝（gfx950 支持），对应硬件的 `buffer_load ... lds` 指令，数据从 global memory 直接进 LDS，不经过 VGPR
- **`ttg.async_commit_group`**: 将前面的 async copy 打包成一个 group，后续通过 `async_wait` 等待完成

#### 3. Main Loop（流水线主体）

```mlir
%result:8 = scf.for %i = %c0 to %N_minus_1 step %c1
    iter_args(%acc, %a_ptrs, %b_ptrs, %buf_idx,
              %a_token, %b_token, %a_buf, %b_buf) {

  // ---- 等待上一次 async copy 完成 ----
  %wait = ttg.async_wait %a_token, %b_token {num = 0 : i32}

  // ---- Stage 0: 为下一次迭代发起 async copy ----
  %a_ptrs_next = tt.addptr %a_ptrs, %cst_64      // 指针步进到下一个 K 块
  %b_ptrs_next = tt.addptr %b_ptrs, %cst_64
  %next_idx = ...                                  // 切换 buffer index: 0→1→0→1
  %a_next_buf = ttg.memdesc_index %a[%next_idx]   // 取另一个 buffer
  %a_copy = ttg.async_copy_global_to_local %a_ptrs_next, %a_next_buf  // 异步加载
  %a_commit = ttg.async_commit_group tokens %a_copy
  // B 同理...

  // ---- Stage 1: 用当前 buffer 的数据做计算 ----
  %a_data = ttg.local_load %a_buf token %wait     // LDS → register
  %b_data = ttg.local_load %b_buf token %wait
  %a_cvt = ttg.convert_layout %a_data             // 转为 dot operand layout
  %b_cvt = ttg.convert_layout %b_data
  %acc_new = tt.dot %a_cvt, %b_cvt, %acc          // MFMA 计算

  scf.yield %acc_new, %a_ptrs_next, %b_ptrs_next, %next_idx,
            %a_commit, %b_commit, %a_next_buf, %b_next_buf
}
```

关键变化：
- **iter_args 从 3 个变成 8 个**：增加了 buffer index、async tokens、memdesc 引用
- **`ttg.async_wait {num = 0}`**: 等待所有未完成的 async copy。`num=0` 表示等到剩余 0 个未完成的 group
- **`ttg.local_load ... token %wait`**: 从 LDS 加载数据到寄存器，依赖 `async_wait` 的完成 token
- 每次迭代同时做两件事：**发起下一批的 async copy**（stage 0）+ **计算当前批的 dot**（stage 1）

#### 4. Epilogue（循环后：处理最后一批）

```mlir
// 等待最后一批 async copy
%wait_final = ttg.async_wait %result#4, %result#5 {num = 0 : i32}

// 从 LDS 加载最后一批数据
%a_last = ttg.local_load %result#6 token %wait_final
%b_last = ttg.local_load %result#7 token %wait_final

// 执行最后一次 dot
%acc_final = scf.if %has_last_iter -> (...) {
    %final = tt.dot %a_last_cvt, %b_last_cvt, %result#0
    scf.yield %final
} else {
    scf.yield %result#0
}

// 释放 shared memory
ttg.local_dealloc %b
ttg.local_dealloc %a
```

Epilogue 用 `scf.if` 保护最后一次 dot，因为当 K 恰好被 BLOCK_K 整除时，循环少跑了一次。

## 时间线对比

### Pipeline 之前（串行）

```
迭代 i:   [====load A,B====][==dot==]
迭代 i+1: [====load A,B====][==dot==]
```

### Pipeline 之后（重叠）

```
Prologue: [====load[0]====]
迭代 0:   [====load[1]====][==dot[0]==]    ← load 和 compute 重叠
迭代 1:   [====load[0]====][==dot[1]==]
...
Epilogue:                   [==dot[N]==]
```

## 新增 Op 总结

| Op | 作用 |
|----|------|
| `ttg.local_alloc` | 分配 shared memory（LDS）|
| `ttg.local_dealloc` | 释放 shared memory |
| `ttg.memdesc_index %buf[%idx]` | 从多 buffer 中按 index 取一个 slice |
| `ttg.async_copy_global_to_local` | 异步 global → LDS 拷贝（硬件 DMA）|
| `ttg.async_commit_group` | 提交 async copy group |
| `ttg.async_wait {num = N}` | 等待直到剩余 ≤ N 个未完成 group |
| `ttg.local_load ... token` | 从 LDS 加载到寄存器，依赖 wait token |
