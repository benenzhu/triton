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
