# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, m: int, n: int, k: int, dtype: str, out_dtype: str) -> Callable:
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, 'tl.disable_warp_specialized': False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _bmm_fp8_func(block_m: int=128, block_n: int=128, block_k: int=64, num_stages: int=3, threads: int=256) -> Callable:
        scale_a_shape = (1,)
        scale_b_shape = (1,)
        k_exact = k % block_k == 0
        n_exact = n % block_n == 0
        m_exact = m % block_m == 0

        @T.prim_func
        def _bmm_fp8_main(a: T.Tensor((batch, m, k), dtype), b: T.Tensor((batch, n, k), dtype), scale_a: T.Tensor(scale_a_shape, 'float32'), scale_b: T.Tensor(scale_b_shape, 'float32'), c: T.Tensor((batch, m, n), out_dtype)) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (bx, by, bz):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_shared = T.alloc_shared((block_m, block_n), out_dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.annotate_layout({a_shared: tilelang.layout.make_swizzled_layout(a_shared), b_shared: tilelang.layout.make_swizzled_layout(b_shared), c_shared: tilelang.layout.make_swizzled_layout(c_shared)})
                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)
                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    if k_exact and m_exact:
                        T.copy(a[bz, m_start:m_start + block_m, k_start:k_start + block_k], a_shared)
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            a_shared[i, j] = T.if_then_else((m_start + i < m) & (k_start + j < k), a[bz, m_start + i, k_start + j], T.cast(0, dtype))
                    if k_exact and n_exact:
                        T.copy(b[bz, n_start:n_start + block_n, k_start:k_start + block_k], b_shared)
                    else:
                        for i, j in T.Parallel(block_n, block_k):
                            b_shared[i, j] = T.if_then_else((n_start + i < n) & (k_start + j < k), b[bz, n_start + i, k_start + j], T.cast(0, dtype))
                    T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_m, block_n):
                    c_local[i, j] = c_local[i, j] * scale_a[0] * scale_b[0]
                T.copy(c_local, c_shared)
                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        c[bz, m_start + i, n_start + j] = c_shared[i, j]
        return _bmm_fp8_main
    return _bmm_fp8_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
