import torch
import triton
from mi355x_gemm import matmul_kernel

M, N, K = 8192, 8192, 8192
flops = 2.0 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

configs = [
    # (num_warps, waves_per_eu, GROUP_SIZE_M, matrix_instr_nonkdim, label)
    (8, 2, 16, 16, 'current'),
    (8, 2, 8,  16, 'grp8'),
    (8, 2, 4,  16, 'grp4'),
    (8, 0, 16, 16, 'wave0'),
    (8, 1, 16, 16, 'wave1'),
    (4, 2, 16, 16, 'warp4'),
    (4, 2, 16, 0,  'w4_m0'),
    (8, 2, 16, 0,  'mfma0'),
    (8, 2, 16, 32, 'mfma32'),
]

for nw, wpe, gsm, mid, label in configs:
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    def run():
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
            GROUP_SIZE_M=gsm, EVEN_K=True, NUM_XCDS=8,
            num_stages=2, num_warps=nw, waves_per_eu=wpe,
            matrix_instr_nonkdim=mid,
        )
        return c

    try:
        run()
        ms = triton.testing.do_bench(run, warmup=50, rep=200)
        print(f'[{label:>8s}] warp={nw} wave={wpe} grp={gsm:>2d} mfma={mid:>2d}  {ms:.3f} ms  {flops/ms/1e9:.0f} TFLOPS')
    except Exception as e:
        print(f'[{label:>8s}] FAILED: {e}')
