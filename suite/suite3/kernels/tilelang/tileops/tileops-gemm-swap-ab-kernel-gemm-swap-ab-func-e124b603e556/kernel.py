# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang
import tilelang.language as T
SWAP_AB_MPAD = 8

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    dtype = pipeguard_bindings['dtype']
    k = pipeguard_bindings['k']
    m = pipeguard_bindings['m']
    n = pipeguard_bindings['n']

    @tilelang.jit(out_idx=[-1], compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_swap_ab_func(block_nn: int=64, block_k: int=128, num_stages: int=4) -> Callable:
        mpad = SWAP_AB_MPAD

        @T.prim_func
        def _gemm_swap_ab_main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k), dtype), c: T.Tensor((m, n), dtype)) -> None:
            with T.Kernel(T.ceildiv(n, block_nn), threads=128) as bx:
                b_smem = T.alloc_shared((block_nn, block_k), dtype)
                a_smem = T.alloc_shared((mpad, block_k), dtype)
                ct_local = T.alloc_fragment((block_nn, mpad), accum_dtype)
                ct_cast = T.alloc_fragment((block_nn, mpad), dtype)
                ct_smem = T.alloc_shared((block_nn, mpad), dtype)
                T.clear(ct_local)
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bx * block_nn, ki * block_k], b_smem)
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct_local, transpose_B=True)
                T.copy(ct_local, ct_cast)
                T.copy(ct_cast, ct_smem)
                for mi, j in T.Parallel(mpad, block_nn):
                    if mi < m and bx * block_nn + j < n:
                        c[mi, bx * block_nn + j] = ct_smem[j, mi]
        return _gemm_swap_ab_main
    return _gemm_swap_ab_func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    return _resolve_kernel(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
