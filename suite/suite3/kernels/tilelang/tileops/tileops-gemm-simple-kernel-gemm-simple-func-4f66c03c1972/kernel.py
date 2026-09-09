# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str='float16') -> Callable:
    """Non-warp-specialized pipelined GEMM for short-mainloop shapes (SM90).

    A stock ``T.Pipelined`` + ``T.gemm`` kernel: every thread cooperates in
    both copy and math, and the compiler schedules the cp.async/TMA overlap.
    On short-K skinny-M NT shapes (the decode-down family: ~16 K iterations,
    about one CTA wave) this beats the warp-specialized kernel by ~4% — with
    so short a mainloop the WS producer warpgroup's fixed costs (barrier
    protocol per iteration, idle tail, 128 threads not doing math) outweigh
    the benefit of its hand-managed deeper ring.

    Selected via config only (``simple: True``, pinned per-shape in
    ``GemmKernel._TUNED_CONFIGS``). Requires tiles that divide the problem
    exactly; the builder raises ``ValueError`` otherwise.
    """
    if trans_a:
        raise ValueError('_gemm_simple_kernel supports trans_a=False only')
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_simple_func(block_m: int=64, block_n: int=128, block_k: int=128, num_stages: int=4, threads: int=128, panel_size: int=8, cluster_m: int=1) -> Callable:
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(f'_gemm_simple_kernel requires exact tiling: got m={m} % {block_m}, n={n} % {block_n}, k={k} % {block_k}')
        if cluster_m > 1:
            if m // block_m % cluster_m:
                raise ValueError(f'cluster_m={cluster_m} must divide the M grid ({m // block_m} tiles)')
            if panel_size > 0:
                raise ValueError('cluster_m > 1 requires panel_size == 0')
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        def _launch():
            if cluster_m > 1:
                return T.ClusterKernel(n // block_n, m // block_m, cluster_dims=(1, cluster_m, 1), threads=threads)
            return T.Kernel(n // block_n, m // block_m, threads=threads)

        @T.prim_func
        def _gemm_simple_main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k) if trans_b else (k, n), dtype), c: T.Tensor((m, n), dtype)) -> None:
            with _launch() as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared(b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.clear(c_local)
                for ki in T.Pipelined(k // block_k, num_stages=num_stages):
                    T.copy(a[by * block_m, ki * block_k], a_smem)
                    if trans_b:
                        T.copy(b[bx * block_n, ki * block_k], b_smem)
                    else:
                        T.copy(b[ki * block_k, bx * block_n], b_smem)
                    T.gemm(a_smem, b_smem, c_local, transpose_B=trans_b)
                T.copy(c_local, c[by * block_m, bx * block_n])
        return _gemm_simple_main
    return _gemm_simple_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
