# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/expand_kernel.py#L6-L31
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

@tilelang.jit
def expand_to_mhc_fwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def expand_to_mhc_fwd_kernel(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xl = T.alloc_fragment((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            o[i, m, j] = xl[ti, tj]

    return expand_to_mhc_fwd_kernel


def _resolve_kernel(**pipeguard_bindings):
    return expand_to_mhc_fwd_tl


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
