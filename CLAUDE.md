# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triton is a language and compiler for writing highly efficient custom deep-learning primitives. It compiles Python-like kernel code into optimized GPU machine code (PTX/AMDGCN) through a multi-stage MLIR-based pipeline. Supported targets: NVIDIA GPUs (Compute Capability 8.0+) and AMD GPUs (ROCm 6.2+).

## Build Commands

```shell
# Install build dependencies
pip install -r python/requirements.txt

# Build and install (editable/dev mode)
pip install -e . --no-build-isolation -v

# Full dev setup (includes test deps + torch)
make dev-install

# Incremental rebuild after C++ changes (faster than pip install)
make all

# Build only triton-opt tool
make triton-opt
```

**Build environment variables:**
- `TRITON_BUILD_WITH_CLANG_LLD=true` - use clang/lld for faster builds
- `TRITON_BUILD_WITH_CCACHE=true` - use ccache
- `MAX_JOBS=N` - limit parallel build jobs (useful if running out of memory)
- `DEBUG=1` - debug build; `TRITON_REL_BUILD_WITH_ASSERTS=1` - release with assertions (default)
- `LLVM_INCLUDE_DIRS`, `LLVM_LIBRARY_DIR`, `LLVM_SYSPATH` - for custom LLVM builds

## Testing

```shell
# All tests (requires GPU)
make test

# Tests without GPU
make test-nogpu

# Individual test categories
make test-lit          # MLIR lit tests (FileCheck-based, no GPU needed)
make test-cpp          # C++ unit tests (no GPU needed)
make test-unit         # Python unit tests (GPU required)
make test-gluon        # Gluon framework tests
make test-regression   # Regression tests
make test-interpret    # Interpreter mode tests (no GPU needed)
make test-proton       # Profiler tests

# Run a single Python test file
python -m pytest python/test/unit/language/test_core.py -x -s

# Run a single lit test
<build_dir>/bin/triton-opt test/TritonGPU/some_test.mlir
```

## Linting and Formatting

Pre-commit hooks handle formatting. Setup: `pip install pre-commit && pre-commit install`

- **Python**: ruff (linting), yapf (formatting, pep8-based, 120 col limit)
- **C++**: clang-format (LLVM style)
- **Type checking**: mypy (limited to files listed in pyproject.toml)

## Architecture

### Compiler Pipeline

Triton kernels go through these IR stages (each is a set of MLIR passes):

1. **Python AST -> TTIR** (`python/triton/compiler/code_generator.py`): Triton's Python DSL is parsed and lowered to Triton IR (TTIR), a high-level MLIR dialect
2. **TTIR -> TTGIR** (`lib/Conversion/TritonToTritonGPU/`): Triton GPU IR adds layout annotations and GPU-specific information
3. **TTGIR -> LLIR** (`lib/Conversion/TritonGPUToLLVM/`): Lowering to LLVM IR via backend-specific passes
4. **LLIR -> PTX/AMDGCN**: LLVM backend code generation
5. **PTX -> cubin / AMDGCN -> hsaco**: Assembly to binary

The `compile()` function in `python/triton/compiler/compiler.py` orchestrates the full pipeline. Each backend registers its stages via `add_stages()`.

### MLIR Dialects (C++ core)

- **Triton dialect** (`include/triton/Dialect/Triton/`, `lib/Dialect/Triton/`): Core ops like `tt.dot`, `tt.load`, `tt.store`, `tt.reduce`
- **TritonGPU dialect** (`include/triton/Dialect/TritonGPU/`, `lib/Dialect/TritonGPU/`): GPU-specific layouts (blocked, MMA, shared memory), pipelining, warp specialization
- **Gluon dialect** (`include/triton/Dialect/Gluon/`, `lib/Dialect/Gluon/`): Newer high-level structured programming model
- **TritonNvidiaGPU dialect** (`include/triton/Dialect/TritonNvidiaGPU/`): NVIDIA-specific ops (TMA, warp groups)
- Dialect definitions use MLIR TableGen (`.td` files in `include/`) to generate C++ boilerplate

### Backend Structure

Backends live under `third_party/` and each provides:
- `backend/compiler.py` - Backend-specific compilation stages and options
- `backend/driver.py` - GPU runtime interaction (memory allocation, kernel launch)
- `backend/driver.c` - C runtime for kernel launching
- `lib/` - Backend-specific MLIR passes and lowerings
- `include/` - Backend-specific dialect definitions
- `language/` - Backend-specific language extensions (e.g., `tl.extra.nvidia`)

**NVIDIA backend** (`third_party/nvidia/`): Targets PTX/cubin, includes Hopper-specific TMA and warp group ops, downloads ptxas/cuobjdump at build time.

**AMD backend** (`third_party/amd/`): Targets AMDGCN/hsaco, has its own dialect (`TritonAMDGPU`) for AMD-specific optimizations.

### Python Layer

- `python/triton/language/` - The `tl` namespace: user-facing kernel language primitives (`core.py` is the main file, `semantic.py` implements type checking and lowering to MLIR ops)
- `python/triton/runtime/jit.py` - `@triton.jit` decorator, kernel launch, autotuning
- `python/triton/runtime/interpreter.py` - CPU interpreter for debugging kernels without a GPU (`TRITON_INTERPRET=1`)
- `python/triton/knobs.py` - All configuration knobs (env vars and programmatic)
- `python/triton/compiler/` - Python-side compiler orchestration
- `python/triton/experimental/gluon/` - Gluon: structured programming model for advanced kernel patterns

### Python Bindings

`python/src/` contains pybind11 bindings (`main.cc`, `ir.cc`, `passes.cc`, etc.) that expose the C++ MLIR infrastructure to Python as `triton._C.libtriton`.

### Key Tools

- `triton-opt` - MLIR pass runner for Triton dialects (used by lit tests)
- `triton-llvm-opt` - LLVM opt wrapper
- `triton-tensor-layout` - Visualize tensor layouts
- `triton-reduce` - MLIR test case reducer

### Debugging Environment Variables

Key variables for debugging (see `python/triton/knobs.py` and README for full list):
- `MLIR_ENABLE_DUMP=1` - dump IR before every MLIR pass
- `TRITON_INTERPRET=1` - run kernels on CPU interpreter (supports Python breakpoints)
- `TRITON_ALWAYS_COMPILE=1` - bypass compilation cache
- `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=<dir>` - dump IR at each stage
- `TRITON_KERNEL_OVERRIDE=1` + `TRITON_OVERRIDE_DIR=<dir>` - override compiled kernels with modified IR
- `TRITON_FRONT_END_DEBUGGING=1` - show full stack traces from compiler frontend

## Code Conventions

- C++17 standard, compiled with `-Werror` and `-fno-exceptions -fno-rtti`
- MLIR dialect definitions use TableGen (`.td` files generate C++ via `mlir-tblgen`)
- Python line length: 120 characters
- Tests: Python tests use pytest; C++ IR tests use LLVM lit/FileCheck

---

## MI355X GEMM Optimization Work

本节记录在 MI355X (gfx950, 256 CU) 上的 bf16 GEMM 优化工作，方便换机器时快速恢复上下文。

### 关键文件

| 文件 | 说明 |
|------|------|
| `mi355x_gemm.py` | 优化后的 GEMM kernel + benchmark（NT layout） |
| `mi355gemm_learn.py` | 用于 pipeline 学习的简化版 kernel |
| `mi355gemm_pingpong_learn.py` | 触发 BlockPingpong pass 的 GEMM kernel（128x128x64, num_stages=3） |
| `annotate_pipeline.py` | 工具脚本：给 MLIR pipeline dump 标注 Python 源码行 |
| `sweep_gemm.py` | 参数扫描脚本 |
| `docs/triton-lower-learning/` | Triton lowering 学习笔记系列 |
| `docs/compiler-explorer-setup.md` | Compiler Explorer 本地搭建指南 |

### 当前最优 GEMM 配置（8192x8192x8192 bf16 NT）

```python
BLOCK_M=256, BLOCK_N=256, BLOCK_K=64
num_warps=8, num_stages=2, waves_per_eu=2
GROUP_SIZE_M=8, matrix_instr_nonkdim=16
EVEN_K=True, NUM_XCDS=8
sanitize_overflow=False  # 可安全关闭，减少编译时间
```

### 性能数据（MI355X gfx950, 256 CU）

| 实现 | TFLOPS | 备注 |
|------|--------|------|
| HipKittens (C++ MFMA) | ~1258 | 纯寄存器 tiling，0 LDS |
| torch.matmul (rocBLAS) | ~1105-1201 | NT 更快 |
| **ours (Triton optimized)** | **~1048** | 和 torch 差距 ~5-13% |
| aiter (Triton) | ~940 | |
| ours (baseline, 无优化) | ~889 | |

### 优化历程

1. **Baseline**: 256x256x64, num_warps=8, num_stages=2 → **889 TFLOPS**
2. **+EVEN_K**: 去掉 K 维度 mask → 减少比较/select 指令
3. **+XCD remap**: 8 XCDs 间均匀分布 workgroup → 减少 L2 竞争
4. **+tl.assume**: 提示编译器 stride > 0
5. **+waves_per_eu=2**: 改善延迟隐藏
6. **+matrix_instr_nonkdim=16**: 选择 16x16 MFMA
7. **+GROUP_SIZE_M=8**: L2 swizzle 调优（16→8 更优）
8. 以上全部 → **1048 TFLOPS**（+18% vs baseline）

### MI355X 硬件特征

- gfx950 (CDNA4), 256 CU, 8 XCD
- Shared memory (LDS): 160 KB/workgroup（MI300X 是 256 KB）
- 256x256x64 bf16 双缓冲 LDS: (256×64 + 64×256) × 2B × 2 = 128 KB < 160 KB ✓
- Warp size: 64 threads
- MFMA: `v_mfma_f32_32x32x16_bf16` 或 `v_mfma_f32_16x16x32_bf16`

### Pingpong Pass 状态

Pingpong scheduling (`third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`) 默认在 gfx950 上开启，但**未在我们的 kernel 上生效**。原因：gfx950 上 async_copy 默认开启，pipeline 后全局加载变成 `AsyncCopyGlobalToLocalOp` 而非 `tt::LoadOp`。BlockPingpong 的通用路径（Four/Two/One PP Clusters）依赖 `gLoadOps`（`tt::LoadOp`），async copy 模式下 `gLoadOps` 为空→被拒绝。gfx950 上唯一能触发的标准 GEMM 路径是 `TwoClusterWithLocalLoadAndAll`，要求 `numStages==3`（非 2）。见 `mi355gemm_pingpong_learn.py`（128x128x64, num_stages=3）已验证触发。

### 外部仓库

| 仓库 | 路径 | 说明 |
|------|------|------|
| aiter | `/root/aiter` | AMD 的 Triton op 库，bf16 GEMM 约 940 TFLOPS |
| HipKittens | `/root/HipKittens` | C++ MFMA 手写 kernel，1258 TFLOPS |
| tritonBLAS | `/root/tritonBLAS` | ROCm GEMM 库（MI300X 为主，MI355X shared memory 不够） |
| compiler-explorer-triton | `/root/compiler-explorer-triton` | Triton Compiler Explorer，已本地部署 |

### Compiler Explorer 本地部署

```bash
# 启动（端口 10240）
cd /root/compiler-explorer-triton && npx tsx ./app.ts --language triton &

# 配置在 etc/config/triton.defaults.properties
# 已改为本地 Triton + gfx950 + warp_size=64
```

已修改 `lib/parsers/mlir-pass-dump-parser.ts` 添加了 Opt Pipeline 视图的 source mapping（解析 `#loc` 定义并关联到源码行）。

### Pipeline 分析工具

```bash
# 方法 1: triton_wrapper（无需 GPU）
python3 /root/compiler-explorer-triton/etc/scripts/triton_wrapper.py \
    kernel.py --output_file /tmp/output.asm \
    --opt_pipeline_file /tmp/pipeline.txt \
    --backend hip --arch gfx950 --warp_size 64

# 方法 2: annotate_pipeline（标注 Python 源码到 IR）
python3 annotate_pipeline.py /tmp/pipeline.txt -o /tmp/pipeline.annotated.txt

# 方法 3: 直接跑 kernel 并 dump
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python3 kernel.py 2>/tmp/dump.txt
```

### Triton Lowering 学习笔记

已完成的文档（`docs/triton-lower-learning/`）：
- `01-python-to-ttir.md` — Python→TTIR：函数签名、constexpr、tl.assume、tensor ops、scf.for、tt.dot
- `02-schedule-and-pipeline.md` — ScheduleLoops 标注 stage/cluster + Pipeline 软件流水线展开（双缓冲 LDS、async copy、prologue/epilogue）
- `03-convert-to-buffer-ops.md` — ConvertToBufferOps：async_copy → buffer_load_to_local（32-bit offset、SGPR base）
- `03a-canonicalize-pointers.md` — CanonicalizePointers：tensor of pointers → FatPtr{scalar base, tensor offset}，uniform/non-uniform 分离

### AMD 后端关键 Pass 顺序

```
TTIR 阶段:
  Inliner → Canonicalizer → TritonCombineOps → CSE → LICM → LoopUnroll

TTGIR 阶段:
  ConvertTritonToTritonGPU → Coalesce → F32DotTC → RemoveLayoutConversions
  → AccelerateMatmul (选择 MFMA 16x16x32) → OptimizeEpilogue
  → OptimizeDotOperands → HoistLayoutConversions → SinkLayoutConversions
  → Canonicalizer → LICM

Pipeline 阶段:
  ScheduleLoops (标注 stage/cluster) → Pipeline (软件流水线)
  → CoalesceAsyncCopy → ConvertToTensorOps → RemoveLayoutConversions
  → ReduceDataDuplication → MoveUpPrologueLoads
  → BlockPingpong (未生效) → CanonicalizePointers → ConvertToBufferOps
  → FoldTrueCmpI → CSE

LLVM 阶段:
  AllocateSharedMemory → ConvertTritonAMDGPUToLLVM → ConvertToLLVM
  → EnableLineInfo → ConvertBuiltinFuncToLLVM
```
