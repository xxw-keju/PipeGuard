# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    dtype = pipeguard_bindings['dtype']
    m = pipeguard_bindings['m']
    n = pipeguard_bindings['n']
    split_k = pipeguard_bindings['split_k']

    @tilelang.jit(compile_flags=['-O3', '-DENABLE_BF16'])
    def _splitk_reduce_func(elems_per_cta: int=1024) -> Callable:

        def _slice_sum(w, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            return expr

        @T.prim_func
        def _splitk_reduce_main(w: T.Tensor((split_k, m, n), accum_dtype), c: T.Tensor((m, n), dtype)) -> None:
            total = m * n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // n
                        gj = idx % n
                        c[gi, gj] = T.cast(_slice_sum(w, gi, gj), dtype)
        return _splitk_reduce_main
    return _splitk_reduce_func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    return _resolve_kernel(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
