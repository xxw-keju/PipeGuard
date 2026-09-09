# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/pre_apply_mix_kernel.py#L57-L110
# The exact upstream excerpt is stored in original_excerpt.py.

import math

import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS, out_idx=[4])
def _mhc_pre_apply_mix_bwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic('n')
    h = hidden
    mhc = mhc_mult

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def _mhc_pre_apply_mix_bwd_kernel(
        o_grad: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        x_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        mix_grad: T.Tensor[(n, mhc), T.float32],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl, disable_tma=True)

            mgl = T.alloc_reducer(mhc, T.float32, replication='all')
            T.fill(mgl, 0)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                ogs = T.alloc_shared(h_blk, T.bfloat16)
                ogl = T.alloc_fragment(h_blk, T.float32)
                T.copy(o_grad[pid_n, i0_h * h_blk], ogs, disable_tma=True)
                T.copy(ogs, ogl, disable_tma=True)

                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                xgs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xgl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x_grad[pid_n, 0, i0_h * h_blk], xgs, disable_tma=True)
                T.copy(xgs, xgl, disable_tma=True)

                for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                    mgl[i_mhc] += ogl[i1_h] * xl[i_mhc, i1_h]
                    xgl[i_mhc, i1_h] += mixl[i_mhc] * ogl[i1_h]

                T.copy(xgl, x_grad[pid_n, 0, i0_h * h_blk], disable_tma=True)

            T.finalize_reducer(mgl)
            T.copy(mgl, mix_grad[pid_n, 0], disable_tma=True)

    return _mhc_pre_apply_mix_bwd_kernel


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_pre_apply_mix_bwd


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
