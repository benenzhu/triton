import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = num_pid_m * num_pid_n

    # XCD remap for MI355X (8 XCDs)
    pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_mn % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    # Swizzle for L2 locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
        GROUP_SIZE_M=8,
        EVEN_K=(K % 64 == 0),
        NUM_XCDS=8,
        num_stages=2,
        num_warps=8,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
    )
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiter", action="store_true", help="benchmark aiter triton gemm")
    args = parser.parse_args()

    M, N, K = 8192, 8192, 8192
    flops = 2.0 * M * N * K

    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

    print(f"=== GEMM {M}x{N}x{K} bf16 on MI355X ===\n")

    # ---- Our optimized kernel ----
    c_ours = matmul(a, b)
    c_ref = torch.matmul(a, b)
    print(f"[triton]  max diff: {(c_ours.float() - c_ref.float()).abs().max().item():.4f}")
    ms = triton.testing.do_bench(lambda: matmul(a, b), warmup=100, rep=500)
    print(f"[triton]  {ms:.3f} ms  {flops/ms/1e9:.1f} TFLOPS")

    # ---- torch.matmul ----
    ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=100, rep=500)
    print(f"[torch]   {ms:.3f} ms  {flops/ms/1e9:.1f} TFLOPS")

    # ---- aiter (optional) ----
    if args.aiter:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
        c_aiter = gemm_a16w16(a, w, dtype=torch.bfloat16)
        c_ref2 = torch.matmul(a, w.T)
        print(f"[aiter]   max diff: {(c_aiter.float() - c_ref2.float()).abs().max().item():.4f}")
        ms = triton.testing.do_bench(lambda: gemm_a16w16(a, w, dtype=torch.bfloat16), warmup=100, rep=500)
        print(f"[aiter]   {ms:.3f} ms  {flops/ms/1e9:.1f} TFLOPS")
