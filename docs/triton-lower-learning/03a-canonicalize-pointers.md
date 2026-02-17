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

## 源码 Walkthrough

源码在 `third_party/amd/lib/TritonAMDGPUTransforms/CanonicalizePointers.cpp`（约 2000 行）。

### 整体架构

pass 使用 MLIR 的 **Dialect Conversion** 框架，把 tensor of pointers 这种"类型"转换为 (scalar base, tensor offset) 这种"类型对"。核心数据结构是 `FatPointers`（line 456），本质是一个 `DenseMap<(Value base, Value offset), FatPtrAttrs>`。

### 入口: `runOnOperation`（line 1919）

```cpp
void TritonAMDGPUCanonicalizePointersPass::runOnOperation() {
    FatPointers fatPtrs;

    // 第 1 步: 初始化函数指针参数
    // 对每个 tt.ptr 参数创建 FatPtr{base=arg, offset=0}
    InitFuncPtrArgs pat(&getContext(), fatPtrs, enableLargeTensorPtrCanon);
    pat.matchAndRewrite(func, rewriter);

    // 第 2 步: 收集需要改写的 op（前向切片）
    // 从每个指针参数出发，追踪所有传递使用
    SetVector<Operation*> opsToRewrite;
    for (auto arg : func.getArguments()) {
        if (!isa<tt::PointerType>(arg.getType())) continue;
        for (auto &use : arg.getUses())
            getForwardSliceImpl(&use, use.getOwner(), &opsToRewrite);
    }

    // 第 3 步: 应用 conversion patterns
    RewritePatternSet patterns;
    patterns.add<ConvertSplatOp, ConvertAddPtrOp, ConvertSCFForOp,
                 MaterializeFatPointer<tt::LoadOp>,
                 MaterializeFatPointer<tt::StoreOp>, ...>(...);
    applyPartialConversion(func, target, patterns);
}
```

### 第 1 步: `InitFuncPtrArgs`（line 1675）

为每个指针参数插入一个 `unrealized_conversion_cast`，建立 FatPtr 的初始映射。

```cpp
// 对于 %A: !tt.ptr<bf16> {tt.pointer_range = 32}
Value zeroOffset = arith.constant 0 : i32;  // 32-bit（因为 pointer_range=32）
auto dummyCast = unrealized_cast(%A, %zeroOffset) -> tt.ptr;
// 所有 %A 的使用者现在看到的是 dummyCast 的结果
// fatPtrs[{%A, %zeroOffset}] = {canNarrow=true, smallTensorBase=%A}
```

`pointer_range=32` 的参数用 32-bit offset（`i32`），否则用 64-bit（`i64`）。
`smallTensorBase` 被设为原始参数 `%A`，后续 `ConvertAddPtrOp` 会使用不同的策略。

### 第 2 步: `getForwardSliceImpl`（line 1862）

从指针参数出发，沿着 use-def chain 向前追踪，收集所有涉及 pointer 的 op。

```
%A (tt.ptr) ──use──> tt.splat ──use──> tt.addptr ──use──> tt.load
                                           │
                                           └──use──> scf.for ──use──> tt.addptr ──use──> ...
```

特殊处理 `scf.for`（追踪 iter_args）、`scf.if`（追踪两个分支的 yield）、`cf.branch`（追踪 block args）。

### 第 3 步: Conversion Patterns

#### `ConvertSplatOp`（line 643）— splat 指针 → 保留 base，splat offset

```
输入:  tt.splat %fatPtr -> tensor<N x !tt.ptr>
       其中 %fatPtr 已被拆为 (base, offset)

输出:  base 不变，offset 被 splat
       result = (base, tt.splat(offset))
```

关键代码（line 669-672）：
```cpp
tt::SplatOp offset = tt::SplatOp::create(rewriter, loc, newOffsetType, fatPtrOffset);
rewriter.replaceOpWithMultiple(splatOp, {{fatPtrBase, offset}});
```

#### `ConvertAddPtrOp`（line 722）— 核心：分解 offset 为 uniform + non-uniform

三种情况（line 765-839）：

**情况 1: 标量指针更新**（line 766）
```cpp
// tt.addptr %scalar_ptr, %scalar_offset → 只更新 base
if (isa<tt::PointerType>(addPtrOp.getPtr().getType())) {
    auto newBase = tt.addptr(fatPtrBase, origOffset);  // 标量加法
    result = (newBase, fatPtrOffset);  // offset 不变
}
```

**情况 2: 常量 tensor offset**（line 783）
```cpp
// 整个 offset 是 splat(constant)，是 uniform 的
if (auto scalarConst = maybeGetOrCreateScalarConstant(origOffset)) {
    auto newBase = tt.addptr(fatPtrBase, scalarConst);  // 合并到 base
    result = (newBase, fatPtrOffset);  // offset 不变
}
```

**情况 3: 通用 tensor offset**（line 797）— 需要分解
```cpp
// 调用 createDecomposeOffsetFromExpr 递归分解
auto [uniformOffset, nonUniformOffset] =
    createDecomposeOffsetFromExpr(rewriter, loc, origOffset, bitness);

// uniform 部分加到 base（标量运算）
auto newBase = tt.addptr(fatPtrBase, uniformOffset);

// non-uniform 部分加到 offset（tensor 运算）
Value newOffset = arith.addi(fatPtrOffset, nonUniformOffset);

result = (newBase, newOffset);
```

对于 `smallTensorBase` 非空的情况（即 `tt.pointer_range=32`），走 `rewriteSmallTensorPtr`（line 843），策略不同：不推进 base，而是把所有 offset 累加在一起保持 32-bit。

#### `createDecomposeOffsetFromExpr`（line 384）— 递归分解 uniform/non-uniform

这是分解算法的核心。递归遍历 offset 表达式树：

```cpp
// Base case 1: splat(scalar) → uniform=scalar, non-uniform=0
if (auto scalarConst = maybeGetOrCreateScalarConstant(expr))
    return {scalarConst, tensorZero};

// Base case 2: block argument (tensor) → uniform=0, non-uniform=expr
if (isa<BlockArgument>(expr))
    return {scalarZero, expr};

// Recursive: 按 op 类型分派
TypeSwitch(expr.getDefiningOp())
    .Case<arith::AddIOp>   → createDecomposeOffsetFromAdd   // (line 334)
    .Case<arith::MulIOp>   → createDecomposeOffsetFromMul   // (line 355)
    .Case<tt::BroadcastOp> → 递归分解 src，broadcast non-uniform 部分
    .Case<tt::ExpandDimsOp>→ 递归分解 src，expand non-uniform 部分
    .Default               → uniform=0, non-uniform=expr   // 无法分解
```

加法的分解（line 334）：
```
decompose(A + B) = { U(A) + U(B),  NU(A) + NU(B) }
```

乘法的分解（line 355）：
```
decompose(A * B) = { U(A) * U(B),  NU(A)*NU(B) + NU(B)*U(A) + U(A)*NU(B) }
```

举例：`offset = splat(pid * 1024) + make_range(0, 1024)`
```
decompose(splat(pid*1024))  = { pid*1024, tensor<0> }    // splat → 全 uniform
decompose(make_range(0,1024)) = { 0, make_range(0,1024) } // make_range → 全 non-uniform
decompose(add) = { pid*1024 + 0, tensor<0> + make_range(0,1024) }
               = { pid*1024,     make_range(0,1024) }
```

#### `MaterializeFatPointer<tt::LoadOp>`（line 1592）— 在 load/store 处重组指针

当遇到 `tt.load` 或 `tt.store` 时，需要把 `(base, offset)` 重新组装成 tensor of pointers：

```cpp
// 调用 createTensorPointer (line 556):
Value ptr = tt.splat(base) : !tt.ptr -> tensor<N x !tt.ptr>
Value result = tt.addptr(ptr, offset) : tensor<N x !tt.ptr>

// 如果 canNarrow=true 且 offset 是 64-bit，先截断到 32-bit
if (fatPtrAttrs.canNarrow && offset.bitwidth > 32)
    offset = arith.trunci(offset, i32);
```

#### `ConvertSCFForOp`（line 1082）— 循环的 iter_args 拆分

对于循环中传递的 tensor of pointers，需要把一个 iter_arg 拆成两个（base + offset）：

```
原始: scf.for ... iter_args(%ptr: tensor<N x !tt.ptr>)
变换: scf.for ... iter_args(%base: !tt.ptr, %offset: tensor<N x i32>)
```

同时更新 `scf.yield` 将一个 yield 值拆成两个。

### 运行测试

```bash
# 运行所有 CanonicalizePointers 测试
<build_dir>/bin/triton-opt test/TritonGPU/amd/amd-canonicalize-pointers.mlir \
    -tritonamdgpu-canonicalize-pointers -canonicalize -verify-diagnostics

# 单独测试某个 case（用 split-input-file）
<build_dir>/bin/triton-opt test/TritonGPU/amd/amd-canonicalize-pointers.mlir \
    -split-input-file -tritonamdgpu-canonicalize-pointers -canonicalize
```
