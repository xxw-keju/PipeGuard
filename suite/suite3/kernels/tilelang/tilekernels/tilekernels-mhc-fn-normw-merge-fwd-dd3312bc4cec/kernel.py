# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/norm_fn_kernel.py#L11-L28
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

@tilelang.jit
def _mhc_fn_normw_merge_fwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_fwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return _mhc_fn_normw_merge_fwd_


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_fn_normw_merge_fwd


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
