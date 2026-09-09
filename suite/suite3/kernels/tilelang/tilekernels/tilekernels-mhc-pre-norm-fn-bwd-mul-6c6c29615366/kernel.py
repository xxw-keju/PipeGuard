# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/norm_fn_kernel.py#L208-L285
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 128,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_mul_kernel(
        # Gradient of output
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Saved inputs
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        # Computed gradient of inputs
        x_grad: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn_grad: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
    ) -> None:
        with T.Kernel(n_rms_group, T.ceildiv(rms_group_size, hidden_block)) as (pid_y, pid_z):
            yz = pid_y * rms_group_size + pid_z * hidden_block

            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            for i, j in T.Parallel(32, hidden_block):
                if i < mhc_mult3:
                    fn_smem[i, j] = fn[i, yz + j]
                else:
                    fn_smem[i, j] = 0

            fn_grad_frag = T.alloc_fragment((32, hidden_block), T.float32)
            T.fill(fn_grad_frag, 0)

            for px in T.serial(T.ceildiv(num_tokens, token_block)):
                x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
                T.copy(x[px * token_block, yz], x_smem)

                padded_grad = T.alloc_shared((token_block, 32), T.float32)
                for i, j in T.Parallel(token_block, 32):
                    if j < mhc_mult3:
                        padded_grad[i, j] = out_mul_grad[px * token_block + i, pid_y, j]
                    else:
                        padded_grad[i, j] = 0

                x_grad_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_grad[px * token_block, yz], x_grad_frag)

                T.gemm(
                    padded_grad,
                    x_smem,
                    fn_grad_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )
                T.gemm(
                    padded_grad,
                    fn_smem,
                    x_grad_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False,
                )

                sqrsum_grad_frag = T.alloc_fragment((token_block, 1), T.float32)
                T.copy(sqrsum_grad[px * token_block, pid_y], sqrsum_grad_frag)
                for i, j in T.Parallel(token_block, hidden_block):
                    x_grad_frag[i, j] += 2 * x_smem[i, j] * sqrsum_grad_frag[i, 0]

                T.copy(x_grad_frag, x_grad[px * token_block, yz])

            T.copy(fn_grad_frag, fn_grad[0, yz])

    return _mhc_pre_norm_fn_bwd_mul_kernel


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_pre_norm_fn_bwd_mul


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
