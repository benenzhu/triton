"""
mi355gemm_pingpong_learn.py

仿照 mi355gemm_learn.py 的简化 GEMM kernel，配置参数使其触发 BlockPingpong pass。

== 为什么原始 kernel (mi355gemm_learn.py) 没有触发 pingpong ==

原始配置: BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=2, num_warps=8

在 gfx950 上，async_copy 默认开启（compiler.py:31-32）。Pipeline pass 会将
tt.load 替换为 ttg.async_copy_global_to_local（直接 DMA 到 LDS）。但 BlockPingpong
的通用路径（Four/Two/One PP Clusters）依赖 gLoadOps（tt::LoadOp），而 async copy
模式下 gLoadOps 为空（全是 asyncCopyOps），导致在 line 1151 被拒绝：
    if (gLoadOps.size() != 2 || lLoadOps.size() != 2) return;

gfx950 上唯一能触发的标准 GEMM 路径是 TwoClusterWithLocalLoadAndAll（line 1084）：
    if (numStages == 3 && dotOps.size() == 1 && dotShape[0] > 64 &&
        dotShape[1] > 64 && (elemWidth == 16 || elemWidth == 8))

== 触发条件 ==

  - gfx950 + async_copy（默认开启）
  - numWarps == 8
  - numStages == 3（关键！不是 2）
  - 1 个 dot op，tile > 64x64，bf16

== LDS 用量 ==

  3 × (128×64 + 64×128) × 2B = 96 KB < 160 KB (MI355X limit) ✓

== 验证方式 ==

  # 方法1: triton_wrapper (无需 GPU)
  python3 /root/compiler-explorer-triton/etc/scripts/triton_wrapper.py \\
      mi355gemm_pingpong_learn.py --output_file /tmp/pp.asm \\
      --opt_pipeline_file /tmp/pp_pipeline.txt \\
      --backend hip --arch gfx950 --warp_size 64

  grep -c 's_setprio' /tmp/pp.asm          # 应该 > 0
  grep 'cond_barrier' /tmp/pp_pipeline.txt  # 应该能找到 amdg.cond_barrier

  # 方法2: 直接运行 (需要 GPU)
  TRITON_ALWAYS_COMPILE=1 python3 mi355gemm_pingpong_learn.py
"""
import sys
sys.argv[6] = "gfx950"
sys.argv[8] = "64"

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_pingpong(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel_pingpong[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # Pingpong 触发配置
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        EVEN_K=(K % 64 == 0),
        num_stages=3,   # 关键: gfx950 async copy 路径需要 3 stages
        num_warps=8,     # 关键: 必须 8 warps
        waves_per_eu=2,
    )
    return c


M, N, K = 8192, 8192, 8192

a = torch.randn(M, K, dtype=torch.bfloat16).cuda()
b = torch.randn(N, K, dtype=torch.bfloat16).cuda()  # NT layout: (N, K)
bt = b.T  # view as (K, N), stride=(1, K)

print(f"=== GEMM {M}x{N}x{K} bf16 NT (Pingpong) on MI355X ===\n")

c_ours = matmul(a, bt)
print("done")
