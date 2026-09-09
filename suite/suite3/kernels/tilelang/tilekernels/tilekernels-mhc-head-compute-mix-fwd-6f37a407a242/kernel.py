# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/head_compute_mix_kernel.py#L10-L33
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def mhc_head_compute_mix_fwd_kernel(
        # Input
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Output
        output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            for i1, j in T.Parallel(token_block_size, mhc_mult):
                i = pid * token_block_size + i1
                if i < num_tokens:
                    output_mix[i, j] = T.sigmoid(input_mix[i, j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps

    return mhc_head_compute_mix_fwd_kernel


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_head_compute_mix_fwd


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
