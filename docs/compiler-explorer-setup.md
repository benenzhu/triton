# Triton Compiler Explorer 本地搭建指南

两种方式查看 Triton 每一步 pass 的 IR 变化：
- **方式一**：直接用 `triton_wrapper.py`（推荐，无需 Node.js）
- **方式二**：搭建完整 Compiler Explorer Web UI

---

## 方式一：triton_wrapper.py（命令行，无需额外依赖）

仓库中的 `etc/scripts/triton_wrapper.py` 可以脱离 Compiler Explorer 独立运行，直接 dump 每一步 pass 的 IR。

### 1. 准备

```bash
git clone https://github.com/ShawnZhong/compiler-explorer-triton.git
cd compiler-explorer-triton
```

依赖只需要当前环境里已有的 `triton` 和 `torch`。

### 2. 编写 kernel 文件

kernel 文件需要在顶层直接调用 `kernel[grid](args)`，不要放在 `if __name__` 里。示例 `test_kernel.py`：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

N = 1024
x = torch.randn(N, device='cuda', dtype=torch.float32)
y = torch.randn(N, device='cuda', dtype=torch.float32)
out = torch.empty(N, device='cuda', dtype=torch.float32)
add_kernel[(N // 256,)](x, y, out, N, BLOCK=256)
```

### 3. 运行

```bash
# dump 汇编 + 所有 pass 的 IR（pipeline.txt）
python3 etc/scripts/triton_wrapper.py test_kernel.py \
    --output_file /tmp/output.asm \
    --opt_pipeline_file /tmp/pipeline.txt \
    --backend hip --arch gfx950 --warp_size 64

# 只 dump 各阶段中间文件（TTIR, TTGIR, LLIR, AMDGCN），不 dump 每个 pass
python3 etc/scripts/triton_wrapper.py test_kernel.py \
    --output_file /tmp/output.asm \
    --backend hip --arch gfx950 --warp_size 64
```

参数说明：
- `--backend hip` / `--backend cuda`：选择 AMD 或 NVIDIA
- `--arch gfx950`：GPU 架构（AMD: gfx942/gfx950/gfx1250, NVIDIA: 80/89/90 等）
- `--warp_size 64`：AMD 用 64，NVIDIA 用 32
- `--opt_pipeline_file`：指定后会通过 `MLIR_ENABLE_DUMP=1` dump 所有 pass 的 IR

### 4. 查看结果

```bash
# 查看所有 pass 名称
grep "^// -----// IR Dump Before" /tmp/pipeline.txt

# 查看某个特定 pass 前后的 IR（例如 BlockPingpong）
grep -A 200 "IR Dump Before TritonAMDGPUBlockPingpong" /tmp/pipeline.txt | head -200

# 查看汇编
cat /tmp/output.asm

# 不加 --opt_pipeline_file 时，各阶段文件在 output.asm 同目录下
ls /tmp/output/  # .ttir, .ttgir, .llir, .amdgcn 等
```

### 5. 原理

`triton_wrapper.py` 的工作原理：
- Mock `CompiledKernel` 和 GPU Driver，让 Triton 只编译不执行
- 使用 PyTorch FakeTensor，不需要真正分配 GPU 内存
- 注册自定义 `CacheManager` 截获编译中间产物
- 设置 `MLIR_ENABLE_DUMP=1` 让 MLIR 在每个 pass 前输出 IR

---

## 方式二：完整 Compiler Explorer Web UI

提供浏览器中的交互式界面，支持 source mapping、pass diff、版本对比等高级功能。

### 1. 安装 Node.js (>=20)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
node --version  # 确认 >= 20
```

### 2. 安装 Triton 到 Compiler Explorer 指定路径

Compiler Explorer 期望 Triton 安装在 `/opt/compiler-explorer/triton/` 下。
可以直接用当前环境的 Python 创建软链：

```bash
mkdir -p /opt/compiler-explorer/triton/local/bin
ln -sf $(which python3) /opt/compiler-explorer/triton/local/bin/python3
```

### 3. 修改配置指向本地 Triton

编辑 `etc/config/triton.defaults.properties`，添加本地编译器：

```properties
# 在 group.triton_amd.compilers 列表开头加上 triton_amd_local
group.triton_amd.compilers=triton_amd_local:triton_amd_340:...
group.triton_amd.options=--backend hip --arch gfx950 --warp_size 64

compiler.triton_amd_local.name=Triton Local (AMD gfx950)
compiler.triton_amd_local.exe=/opt/compiler-explorer/triton/local/bin/python3
```

### 4. 构建并运行

```bash
cd compiler-explorer-triton
make EXTRA_ARGS='--language triton' dev
```

### 5. 打开浏览器

访问 `http://localhost:10242/`

功能：
- **Device Viewer**：查看 TTIR → TTGIR → LLIR → AMDGCN 各阶段 IR
- **Opt Pipeline**：查看每个 MLIR pass 的 IR diff（绿色高亮变化部分）
- **Diff View**：对比不同版本/不同代码的编译产物
- **Source Mapping**：Python 源码到 IR/汇编的对应关系
