# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/sinkhorn_kernel.py#L10-L58
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_sinkhorn_fwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def mhc_sinkhorn_kernel(
        comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
            comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
            row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

            T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            T.reduce_max(comb_frag, row_max, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
            T.reduce_sum(comb_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=1)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            for _ in T.serial(repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            T.copy(comb_frag, comb_res_mix_out[pid_x * token_block_size, 0, 0])

    return mhc_sinkhorn_kernel


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_sinkhorn_fwd


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
