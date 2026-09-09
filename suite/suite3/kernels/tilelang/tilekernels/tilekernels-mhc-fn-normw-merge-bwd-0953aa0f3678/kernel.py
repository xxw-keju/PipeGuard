# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/norm_fn_kernel.py#L31-L61
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

@tilelang.jit
def _mhc_fn_normw_merge_bwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_frag = T.alloc_fragment(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_frag)

            normw_grad_frag = T.alloc_fragment(n_blk, dtype)
            T.clear(normw_grad_frag)

            for i_m in T.serial(m):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                        normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] += normw_grad_frag[i1_n]

    return _mhc_fn_normw_merge_bwd_


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_fn_normw_merge_bwd


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
