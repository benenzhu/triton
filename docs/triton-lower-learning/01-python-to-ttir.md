# Triton Lowering 学习笔记 1: Python → TTIR

本文以一个优化过的 MI355X bf16 GEMM kernel 为例，解析 Triton 如何将 Python 代码 lower 为 TTIR（Triton IR）。

TTIR 是 Triton 编译管线的第一个 MLIR 表示，位于 `ConvertTritonToTritonGPU` pass 之前。此阶段的 IR 是硬件无关的，不包含任何 GPU 特定信息。

## 环境

- GPU: AMD MI355X (gfx950)
- 数据类型: bf16
- Tile: 256×256×64, num_warps=8, num_stages=2

## 生成方式

```bash
# 生成 pipeline dump
python3 compiler-explorer-triton/etc/scripts/triton_wrapper.py \
    test_kernel.py --output_file /tmp/output.asm \
    --opt_pipeline_file /tmp/pipeline.txt \
    --backend hip --arch gfx950 --warp_size 64

# 标注源码
python3 annotate_pipeline.py /tmp/pipeline.txt
```

查看 `ConvertTritonToTritonGPU` pass 之前的 IR 即为最终 TTIR。

## 函数签名

```python
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, ...):
```

lowering 为：

```mlir
tt.func public @matmul_kernel(
    %A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
    %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
    %C: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
    %M: i32 {tt.divisibility = 16 : i32},
    %N: i32 {tt.divisibility = 16 : i32},
    %K: i32 {tt.divisibility = 16 : i32},
    %stride_am: i32 {tt.divisibility = 16 : i32},
    %stride_bn: i32 {tt.divisibility = 16 : i32},
    %stride_cm: i32 {tt.divisibility = 16 : i32}
)
```

要点：
- `tl.constexpr` 参数（BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, NUM_XCDS 等）**不出现在函数签名中**，它们在 Python 前端被内联为常量
- 指针类型为 `!tt.ptr<bf16>`，附带 `tt.divisibility = 16`（来自 PyTorch tensor 的 16 字节对齐）
- 所有 stride 里 `stride_ak=1, stride_bk=1, stride_cn=1` 因为是连续维度，前端直接替换为常量 `1`，不再作为参数传入
- `tt.pointer_range = 32` 表示指针偏移用 32 位整数即可

## tl.assume → llvm.intr.assume

```python
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)  # stride_ak=1, constexpr
```

```mlir
%0 = arith.cmpi sgt, %stride_am, %c0_i32 : i32    // stride_am > 0
llvm.intr.assume %0 : i1                            // 告诉编译器这是恒真的

llvm.intr.assume %true : i1                         // stride_ak=1 > 0, 编译期已知为 true
```

`tl.assume` 直接变成 `llvm.intr.assume`。当 stride 是 constexpr 时（如 `stride_ak=1`），比较结果编译期已知为 `true`。

## tl.program_id → tt.get_program_id

```python
pid = tl.program_id(0)
```

```mlir
%pid = tt.get_program_id x : i32
```

一对一映射。`x` 对应 axis=0。

## tl.cdiv → 整数除法

```python
num_pid_m = tl.cdiv(M, BLOCK_M)  # BLOCK_M=256 是 constexpr
```

```mlir
%num_pid_m = arith.addi %M, %c255_i32 : i32      // M + 255
%num_pid_m_1 = arith.divsi %num_pid_m, %c256_i32  // (M + 255) / 256
```

`tl.cdiv(a, b)` 被内联展开为 `(a + b - 1) / b`（向上取整除法）。`BLOCK_M=256` 作为 constexpr 直接变成常量。

## 标量算术

```python
grid_mn = num_pid_m * num_pid_n
pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS  # NUM_XCDS=8
tall_xcds = grid_mn % NUM_XCDS
tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
xcd = pid % NUM_XCDS
local_pid = pid // NUM_XCDS
```

```mlir
%grid_mn = arith.muli %num_pid_m_1, %num_pid_n_2 : i32
%pids_per_xcd_3 = arith.addi %grid_mn, %c7_i32 : i32        // grid_mn + 7 (=NUM_XCDS-1)
%pids_per_xcd_4 = arith.divsi %pids_per_xcd_3, %c8_i32      // / 8
%tall_xcds = arith.remsi %grid_mn, %c8_i32 : i32             // grid_mn % 8
%tall_xcds_5 = arith.cmpi eq, %tall_xcds, %c0_i32            // == 0 ?
%tall_xcds_6 = arith.select %tall_xcds_5, %c8_i32, %tall_xcds // 三元选择
%xcd = arith.remsi %pid, %c8_i32 : i32
%local_pid = arith.divsi %pid, %c8_i32 : i32
```

标量运算是直接一对一的映射：`*` → `arith.muli`，`//` → `arith.divsi`，`%` → `arith.remsi`，`if-else` → `arith.select` 或 `scf.if`。

## if-else → scf.if

```python
if xcd < tall_xcds:
    pid = xcd * pids_per_xcd + local_pid
else:
    pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
```

```mlir
%2 = arith.cmpi slt, %xcd, %tall_xcds_6 : i32
%3 = scf.if %2 -> (i32) {
    %pid_50 = arith.muli %xcd, %pids_per_xcd_4 : i32
    %pid_51 = arith.addi %pid_50, %local_pid : i32
    scf.yield %pid_51 : i32
} else {
    %pid_50 = arith.muli %tall_xcds_6, %pids_per_xcd_4 : i32
    %pid_51 = arith.subi %xcd, %tall_xcds_6 : i32
    %pid_52 = arith.subi %pids_per_xcd_4, %c1_i32 : i32
    %pid_53 = arith.muli %pid_51, %pid_52 : i32
    %pid_54 = arith.addi %pid_50, %pid_53 : i32
    %pid_55 = arith.addi %pid_54, %local_pid : i32
    scf.yield %pid_55 : i32
}
```

Python 的 `if-else` 变成 `scf.if`（Structured Control Flow），两个分支通过 `scf.yield` 返回结果。注意这是**值语义**，`%3` 接收 if/else 的返回值。

## Tensor 操作: tl.arange, 广播, 指针算术

```python
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
```

```mlir
%offs_am = arith.muli %pid_m_9, %c256_i32 : i32                         // pid_m * 256 (标量)
%offs_am_10 = tt.make_range {end = 256 : i32, start = 0 : i32}          // [0,1,...,255] (tensor<256xi32>)
%offs_am_11 = tt.splat %offs_am : i32 -> tensor<256xi32>                // 标量广播为向量
%offs_am_12 = arith.addi %offs_am_11, %offs_am_10 : tensor<256xi32>     // 逐元素加
%offs_am_13 = tt.splat %M : i32 -> tensor<256xi32>                      // M 广播为向量
%offs_am_14 = arith.remsi %offs_am_12, %offs_am_13 : tensor<256xi32>    // 逐元素取模
```

关键 op 类型：
- **`tt.make_range`**: 对应 `tl.arange(start, end)`，生成一维整数向量
- **`tt.splat`**: 标量广播为 tensor（MLIR 不允许标量和 tensor 直接运算）
- **`arith.addi/muli/remsi`**: 当操作数是 tensor 类型时，自动变成逐元素操作

## 指针构造: expand_dims + broadcast + addptr

```python
a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
```

```mlir
// offs_am[:, None] — 从 tensor<256xi32> 变成 tensor<256x1xi32>
%a_ptrs = tt.expand_dims %offs_am_14 {axis = 1 : i32}
    : tensor<256xi32> -> tensor<256x1xi32>

// * stride_am
%a_ptrs_19 = tt.splat %stride_am : i32 -> tensor<256x1xi32>
%a_ptrs_20 = arith.muli %a_ptrs, %a_ptrs_19 : tensor<256x1xi32>

// offs_k[None, :] — 从 tensor<64xi32> 变成 tensor<1x64xi32>
%a_ptrs_21 = tt.expand_dims %offs_k {axis = 0 : i32}
    : tensor<64xi32> -> tensor<1x64xi32>

// 广播到 tensor<256x64xi32> 并相加
%a_ptrs_22 = tt.broadcast %a_ptrs_20 : tensor<256x1xi32> -> tensor<256x64xi32>
%a_ptrs_23 = tt.broadcast %a_ptrs_21 : tensor<1x64xi32> -> tensor<256x64xi32>
%a_ptrs_24 = arith.addi %a_ptrs_22, %a_ptrs_23 : tensor<256x64xi32>

// A + offsets — 基指针加偏移得到 tensor of pointers
%a_ptrs_25 = tt.splat %A : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
%a_ptrs_26 = tt.addptr %a_ptrs_25, %a_ptrs_24
    : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
```

Triton 的核心抽象：**tensor of pointers**。`tt.addptr` 将基指针加上偏移量 tensor，得到一个 `tensor<256x64x!tt.ptr<bf16>>`，后续 `tt.load` 直接使用。

`[:, None]` 对应 `tt.expand_dims`，NumPy 风格的广播对应 `tt.broadcast`。

## 主循环: for → scf.for

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc = tl.dot(a, b, acc, input_precision="ieee")
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

```mlir
// tl.cdiv(K, 64) = (K + 63) / 64
%4 = arith.addi %K, %c63_i32 : i32
%5 = arith.divsi %4, %c64_i32 : i32

// scf.for 带 iter_args: 循环变量是 (a_ptrs, b_ptrs, acc)
%acc_35:3 = scf.for %k = %c0_i32 to %5 step %c1_i32
    iter_args(%a_ptrs_50 = %a_ptrs_26,
              %b_ptrs_51 = %b_ptrs_34,
              %acc_52 = %acc)
    -> (tensor<256x64x!tt.ptr<bf16>>,
        tensor<64x256x!tt.ptr<bf16>>,
        tensor<256x256xf32>) : i32 {

    // tl.load — 无 mask (EVEN_K=True)
    %a = tt.load %a_ptrs_50 : tensor<256x64x!tt.ptr<bf16>>
    %b = tt.load %b_ptrs_51 : tensor<64x256x!tt.ptr<bf16>>

    // tl.dot
    %acc_53 = tt.dot %a, %b, %acc_52
        : tensor<256x64xbf16> * tensor<64x256xbf16> -> tensor<256x256xf32>

    // a_ptrs += 64 (BLOCK_K * stride_ak, stride_ak=1)
    %a_ptrs_54 = tt.addptr %a_ptrs_50, %cst_0
        : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
    // b_ptrs += 64 * stride_bn
    %b_ptrs_55 = tt.addptr %b_ptrs_51, %cst
        : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>

    scf.yield %a_ptrs_54, %b_ptrs_55, %acc_53 : ...
}
```

要点：
- `scf.for` 是函数式的：循环变量通过 `iter_args` 传入，`scf.yield` 传出。没有可变状态。
- `tt.load` 在 EVEN_K=True 时没有 mask 参数
- `tt.dot` 是矩阵乘法的核心 op，输入 bf16 输出 f32（累加器精度）
- 指针步进用 `tt.addptr` 加常量 tensor `%cst_0`（dense<64>）

## Epilogue: 类型转换 + Store

```python
c = acc.to(tl.bfloat16)
offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=mask)
```

```mlir
// f32 → bf16 截断
%c = arith.truncf %acc_35#2 : tensor<256x256xf32> to tensor<256x256xbf16>

// 构造输出指针 (同 a_ptrs 的模式: expand_dims + broadcast + addptr)
%c_ptrs = tt.expand_dims %offs_am_12 {axis = 1} : ... -> tensor<256x1xi32>
%c_ptrs_36 = tt.splat %stride_cm : i32 -> tensor<256x1xi32>
%c_ptrs_37 = arith.muli %c_ptrs, %c_ptrs_36 : tensor<256x1xi32>
// ... broadcast to 256x256, addptr with C base ...

// mask: (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
%mask_44 = arith.cmpi slt, %c_ptrs, %mask : tensor<256x1xi32>    // < M
%mask_46 = arith.cmpi slt, %c_ptrs_40, %mask_45 : tensor<1x256xi32>  // < N
%mask_47 = tt.broadcast %mask_44 : tensor<256x1xi1> -> tensor<256x256xi1>
%mask_48 = tt.broadcast %mask_46 : tensor<1x256xi1> -> tensor<256x256xi1>
%mask_49 = arith.andi %mask_47, %mask_48 : tensor<256x256xi1>   // &

// masked store
tt.store %c_ptrs_43, %c, %mask_49 : tensor<256x256x!tt.ptr<bf16>>
```

Epilogue 的 mask 构造模式：两个 1D 比较结果分别 broadcast 到 2D 再做 `andi`。

## TTIR Op 速查

| Python | TTIR Op | 说明 |
|--------|---------|------|
| `tl.program_id(0)` | `tt.get_program_id x` | 获取 workgroup id |
| `tl.arange(0, N)` | `tt.make_range {start=0, end=N}` | 生成 [0..N-1] |
| `tl.load(ptrs)` | `tt.load %ptrs` | tensor of pointers load |
| `tl.store(ptrs, val, mask)` | `tt.store %ptrs, %val, %mask` | masked store |
| `tl.dot(a, b, acc)` | `tt.dot %a, %b, %acc` | 矩阵乘 |
| `tl.cdiv(a, b)` | `arith.addi + arith.divsi` | 内联为 (a+b-1)/b |
| `tl.assume(x)` | `llvm.intr.assume` | 编译器 hint |
| `x[:, None]` | `tt.expand_dims {axis=1}` | 增加维度 |
| 标量→tensor | `tt.splat` | 广播 |
| tensor broadcast | `tt.broadcast` | 形状广播 |
| `ptr + offset` | `tt.addptr` | 指针加偏移 |
| `.to(bf16)` | `arith.truncf` | 类型截断 |
| `if-else` | `scf.if ... scf.yield` | 结构化控制流 |
| `for` | `scf.for ... iter_args ... scf.yield` | 函数式循环 |
| `+` `-` `*` `//` `%` | `arith.addi/subi/muli/divsi/remsi` | 整数算术 |
| `<` `==` | `arith.cmpi slt/eq` | 整数比较 |
| `&` | `arith.andi` | 按位与 |
| `min()` | `arith.minsi` | 有符号最小值 |
| `x if cond else y` | `arith.select` | 三元选择 |

## 下一步

下一篇将分析 TTIR → TTGIR 的变换（`ConvertTritonToTritonGPU` pass），包括：
- Layout 注解的引入
- MFMA 指令的选择（`TritonAMDGPUAccelerateMatmul`）
- 软件流水线（`TritonAMDGPUPipeline`）
