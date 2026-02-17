# Triton Lowering 学习笔记 3a: CanonicalizePointers

`CanonicalizePointers` 是 `ConvertToBufferOps` 的前置 pass，负责将 tensor of pointers 拆分为 **scalar base pointer + tensor offset** 的形式。这是 buffer 指令的前提条件。

## 源码位置

| 文件 | 说明 |
|------|------|
| `third_party/amd/lib/TritonAMDGPUTransforms/CanonicalizePointers.cpp` | pass 实现（~1400行） |
| `test/TritonGPU/amd/amd-canonicalize-pointers.mlir` | 测试用例（1543行，覆盖各种场景） |

## 核心思想：FatPtr

pass 内部维护一个 `FatPtr{basePtr, offset}` 数据结构，将每个 pointer value 分解为：

- **basePtr（标量）**：所有 thread/lane 共享的基地址，存在 SGPR 中
- **offset（tensor）**：每个 lane 各自的偏移量，存在 VGPR 中

算法从函数参数开始追踪：
1. 初始状态：`FatPtr{basePtr=arg0, offset=0}`
2. 遇到 `tt.addptr(ptr, offset)` 时，分离 offset 中的 uniform 部分（标量加到 base）和 non-uniform 部分（加到 tensor offset）
3. 遇到 `tt.load/store` 时，重新组装为 `splat(base) + offset` 形式

## 测试用例解析

### Case 1: 纯 uniform offset（最简单）

```python
# Python 语义: ptr[pid * 1024 : pid * 1024 + 1024]
ptr = splat(arg0) + splat(pid * 1024)  # offset 全是 uniform（标量广播）
load(ptr)
```

变换前：
```mlir
%2 = tt.splat %1 : i32 -> tensor<1024xi32>           // uniform offset → tensor
%3 = tt.splat %arg0 : !tt.ptr -> tensor<1024x!tt.ptr> // base → tensor of pointers
%4 = tt.addptr %3, %2                                 // 构造 tensor of pointers
%5 = tt.load %4                                       // 每个 lane 持有独立 64-bit 指针
```

变换后：
```mlir
%4 = tt.addptr %arg0, %3 : !tt.ptr, i32               // 标量 base + 标量 offset → 新标量 base
%5 = tt.splat %4 : !tt.ptr -> tensor<1024x!tt.ptr>    // splat 新 base
%6 = tt.load %5                                       // 所有 lane 读同一个 base（offset=0）
```

**效果**：uniform 部分被提升到标量 `tt.addptr`，消除了 tensor 级的指针运算。

### Case 2: uniform + non-uniform offset（典型 GEMM 场景）

```python
# Python 语义: ptr[pid * 1024 + arange(0, 1024)]
offset = splat(pid * 1024) + arange(0, 1024)  # uniform + non-uniform
ptr = splat(arg0) + offset
load(ptr)
```

变换前：
```mlir
%4 = arith.addi %3, %2 : tensor<1024xi32>            // uniform + non-uniform 混在 tensor 里
%5 = tt.splat %arg0 : !tt.ptr -> tensor<1024x!tt.ptr>
%6 = tt.addptr %5, %4                                 // 64-bit tensor of pointers
%7 = tt.load %6
```

变换后：
```mlir
// uniform 部分提升到标量
%5 = tt.addptr %arg0, %3 : !tt.ptr, i32               // base + uniform_offset → 新 base（标量）

// non-uniform 部分保留为 tensor
%6 = tt.splat %5 : !tt.ptr -> tensor<1024x!tt.ptr>    // splat 新 base
%7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr>, tensor<1024xi32>  // + non-uniform offset
%8 = tt.load %7
```

**效果**：`pid * 1024` 这个 uniform 偏移被合并进 base pointer（标量运算），只有 `arange(0, 1024)` 这个 non-uniform 部分保留在 tensor offset 中。

### Case 3: 多次 addptr + pointer_range=32（GEMM 循环场景）

```python
# 两次 addptr：初始偏移 + 循环步进
offset = splat(pid * 1024) + arange(0, 1024)
ptr1 = splat(arg0) + offset   # 第一次 addptr
ptr2 = ptr1 + offset          # 第二次 addptr（模拟循环步进）
load(ptr2)
```

当 `arg0` **没有** `tt.pointer_range=32` 时：
```mlir
// 在第一次 addptr 处就把 uniform 合并进 base，第二次再合并
// base 会被多次推进
%5 = tt.addptr %arg0, %3 : !tt.ptr, i32     // base + uniform
%7 = tt.addptr %5, %3 : !tt.ptr, i32        // base + uniform 再一次
// offset 需要 64-bit（因为累加可能溢出 32-bit）
%9 = arith.addi %8, %6 : tensor<1024xi64>   // 两次 non-uniform offset 合并（i64）
```

当 `arg0` **有** `tt.pointer_range=32` 时：
```mlir
// 知道总地址范围 < 2GB，可以安全用 32-bit offset
// 不推进 base，而是把所有 offset 累加在一起
%PID_x1024_x2 = arith.addi %PID_x1024, %PID_x1024 : i32  // uniform 合并
%RANGE_x2 = arith.addi %MK_RANGE, %MK_RANGE : tensor<1024xi32>  // non-uniform 合并
%SPLAT = tt.splat %PID_x1024_x2 : i32 -> tensor<1024xi32>
%OFST = arith.addi %SPLAT, %RANGE_x2 : tensor<1024xi32>   // 全部用 i32！
%BASEPTR = tt.splat %arg0 : !tt.ptr -> tensor<1024x!tt.ptr>
%ADDR = tt.addptr %BASEPTR, %OFST : tensor<1024x!tt.ptr>, tensor<1024xi32>
tt.load %ADDR
```

**效果**：有 `pointer_range=32` 时，pass 尽量保留原始 base pointer 不动，把所有偏移累加到一个 32-bit tensor offset 中。这对 `ConvertToBufferOps` 最有利——buffer 指令需要的正是原始 base + 32-bit offset。

### Case 4: 循环中的指针步进

```mlir
// 变换前：循环传递 tensor of pointers
scf.for ... iter_args(%ptr = %initial_ptr) {
    %next = tt.addptr %ptr, %step : tensor<1024x!tt.ptr>, tensor<1024xi32>
    %data = tt.load %next
    scf.yield %next
}

// 变换后：循环传递 base + offset 分开
scf.for ... iter_args(%base = %initial_base, %offset = %initial_offset) {
    %new_offset = arith.addi %offset, %step : tensor<1024xi32>
    %ptr = tt.splat %base → addptr(..., %new_offset)
    %data = tt.load %ptr
    scf.yield %base, %new_offset
}
```

循环中 tensor of pointers 被拆成标量 base（不变）+ tensor offset（每次迭代步进），消除了循环中 tensor 级指针运算的开销。

## 在我们 GEMM 中的效果

变换前（Pipeline 之后）：
```mlir
// 循环体中每次迭代需要重建 tensor of pointers
%a_89 = tt.splat %A : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>   // splat base
%a_90 = tt.addptr %a_89, %a_ptrs_83 : tensor<256x64x!tt.ptr<bf16>>   // + offset tensor
%a_91 = ttg.async_copy_global_to_local %a_90, ...                     // 用 tensor of pointers load
```

变换后：
```mlir
// 直接用 base pointer + offset tensor，无需 splat + addptr
%a_85 = amdg.buffer_load_to_local %A[%a_ptrs_79] ...    // ConvertToBufferOps 可以直接转换
```

CanonicalizePointers 消除了 `tt.splat + tt.addptr` 构造 tensor of pointers 的中间步骤，为后续 `ConvertToBufferOps` 识别 `base + offset` 模式铺平了道路。

## Uniform vs Non-uniform 的判定

```
offset = splat(pid * 1024) + arange(0, 1024)
         ─────────────────   ────────────────
         uniform（标量广播）    non-uniform（每个 lane 不同）
```

- **Uniform**：来自 `tt.splat` 的值——所有 lane 相同，可以提升为标量运算
- **Non-uniform**：来自 `tt.make_range`、`tt.load`（gather）等——每个 lane 不同，必须保留为 tensor

pass 通过追踪值的定义链来判定 uniform/non-uniform。`tt.splat` 的输出是 uniform，`tt.make_range` 的输出是 non-uniform，`arith.addi(uniform, non-uniform)` 的结果是 non-uniform。
