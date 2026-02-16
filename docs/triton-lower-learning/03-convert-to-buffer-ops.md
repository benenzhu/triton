# Triton Lowering 学习笔记 3: ConvertToBufferOps

本文解析 `TritonAMDGPUConvertToBufferOps` pass，它将通用的内存访问指令替换为 AMD 特有的 buffer 指令，是 AMD 后端性能优化的关键步骤之一。

## 源码位置

| 文件 | 说明 |
|------|------|
| `third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp` | pass 实现 |
| `third_party/amd/lib/TritonAMDGPUTransforms/CanonicalizePointers.cpp` | 前置 pass，拆分指针为 base + offset |
| `third_party/amd/backend/compiler.py:258-266` | 注册位置 |

```python
# compiler.py 中调用顺序
amd.passes.ttgpuir.add_canonicalize_pointers(pm)        # 1. 指针规范化
passes.common.add_canonicalizer(pm)                       # 2. 清理
amd.passes.ttgpuir.add_convert_to_buffer_ops(pm, ...)   # 3. 转换为 buffer ops
```

## 前置 Pass: CanonicalizePointers

在 ConvertToBufferOps 之前，`CanonicalizePointers` 先将 `tt.addptr(splat(base_ptr), offsets)` 形式的指针拆分为：
- **base pointer**（标量）：`%A: !tt.ptr<bf16>`
- **offset tensor**（tensor of i32）：`%a_ptrs_26: tensor<256x64xi32>`

这种 base + offset 的分离是 buffer 指令的前提——buffer 指令需要一个统一的基地址和每个 lane 的偏移量。

## 核心变换

### 变换前：`ttg.async_copy_global_to_local`

```mlir
// 先构造 tensor of pointers
%a_37 = tt.splat %A : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>, #linear1>
%a_38 = tt.addptr %a_37, %a_ptrs_26 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
// 然后用 tensor of pointers 做 async copy
%a_39 = ttg.async_copy_global_to_local %a_38, %a_35 mask %a_36
    : tensor<256x64x!tt.ptr<bf16>, #linear1> -> <256x64xbf16, #shared, #smem, mutable>
```

这里有两步：先 splat + addptr 构造 tensor of pointers，再整体做 async copy。每个 thread 持有一个独立的 64 位指针。

### 变换后：`amdg.buffer_load_to_local`

```mlir
// 直接用 base pointer + offset tensor，无需构造 tensor of pointers
%a_37 = amdg.buffer_load_to_local %A[%a_ptrs_26] mask = %a_36 stride = %stride_am
    into %a_35 {contiguity = 8 : i32}
    : <bf16>[tensor<256x64xi32, #linear1>] -> <256x64xbf16, #shared, #smem, mutable>
```

关键变化：
1. **`ttg.async_copy_global_to_local` → `amdg.buffer_load_to_local`**
2. **消除了 `tt.splat` + `tt.addptr`**：不再构造 tensor of pointers，直接传 base 指针 `%A` 和 offset tensor `%a_ptrs_26`
3. **新增 `stride` 参数**：传入 stride 帮助硬件做地址计算
4. **新增 `contiguity = 8`**：告诉硬件连续读取 8 个元素（bf16×8 = 128 bits = dwordx4）

同样的变换也发生在 B 矩阵和 epilogue 的 store 上：

```mlir
// B 矩阵 load：同样的 pattern
%b_41 = amdg.buffer_load_to_local %B[%b_ptrs_33] mask = %b_40 stride = %stride_bn
    into %b_39 {contiguity = 8 : i32}
    : <bf16>[tensor<64x256xi32, #linear>] -> <64x256xbf16, #shared1, #smem, mutable>

// Store C：tt.store 也可能被转为 amdg.buffer_store（取决于条件）
```

## 为什么 buffer 指令更快

### Flat 指令（变换前）

```
每个 lane 持有独立的 64-bit 指针
  → 需要 2 个 VGPR 存地址（高 32 位 + 低 32 位）
  → 硬件需要逐个检查每个 lane 的地址所属的内存段
  → 通过通用的 flat 地址空间访问
```

### Buffer 指令（变换后）

```
所有 lane 共享一个 base pointer（存在 SGPR/descriptor 中）
每个 lane 只需要 32-bit offset（1 个 VGPR）
  → VGPR 使用减半
  → 硬件知道地址一定在 global memory，跳过段检查
  → 走专用的 buffer 访问路径，延迟更低
  → 支持 hardware bounds checking
```

### 汇编层面的对应

```asm
; flat 指令（变换前可能生成的）
flat_load_dwordx4 v[0:3], v[4:5]          ; 需要 64-bit 地址 (2 VGPRs)

; buffer 指令（变换后生成的）
buffer_load_dwordx4 v[0:3], s[0:3], 0 offen  ; base 在 SGPR，offset 在 VGPR (32-bit)
```

## 触发条件

ConvertToBufferOps 不是无条件转换的。在 `canUseBufferOps()` 中检查：

1. **指针必须是 uniform base + non-uniform offset 的形式**（由 CanonicalizePointers 保证）
2. **offset 必须是 32-bit**（i32），64-bit offset 不转换
3. **base pointer 有 `tt.pointer_range = 32` 属性**，或者 offset 可证明 < 2GB
4. **`knobs.amd.use_buffer_ops` 为 true**（默认 true）

```python
# compiler.py 中的条件
if knobs.amd.use_buffer_ops:
    amd.passes.ttgpuir.add_canonicalize_pointers(pm)
    passes.common.add_canonicalizer(pm)
    amd.passes.ttgpuir.add_convert_to_buffer_ops(
        pm, options.arch,
        knobs.amd.use_buffer_atomics,
        knobs.amd.buffer_ops_analyze_small_tensor_range,
    )
```

## 循环体中的变化对比

### 变换前（Pipeline 之后的 async_copy）

```mlir
// 循环体中：构造指针 → async copy
%a_89 = tt.splat %A : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
%a_90 = tt.addptr %a_89, %a_ptrs_83 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
%a_91 = ttg.async_copy_global_to_local %a_90, %a_88 : tensor<256x64x!tt.ptr<bf16>> -> ...
```

### 变换后（buffer_load_to_local）

```mlir
// 循环体中：直接 base + offset
%a_85 = amdg.buffer_load_to_local %A[%a_ptrs_79] into %a_84 {contiguity = 8 : i32}
    : <bf16>[tensor<256x64xi32>] -> ...
```

循环体少了 `tt.splat` 和 `tt.addptr`，减少了指令数量。offset 的步进直接用 `arith.addi`（tensor 加常量 64），不需要重建 tensor of pointers。

## 新增 Op

| Op | 作用 | 对应汇编 |
|----|------|----------|
| `amdg.buffer_load_to_local` | base ptr + offset → LDS（异步 DMA） | `buffer_load_dwordx4 ... lds` |
| `amdg.buffer_store` | base ptr + offset ← 寄存器 | `buffer_store_dwordx4` |
| `amdg.buffer_load` | base ptr + offset → 寄存器 | `buffer_load_dwordx4` |

这些 op 属于 AMD Gluon dialect（`amdg`），在后续 `ConvertTritonAMDGPUToLLVM` pass 中被 lower 为实际的 buffer 指令。
