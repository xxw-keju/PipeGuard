@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_split_mixes_fwd(
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    dtype: T.dtype = T.float32,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    @T.prim_func
    def mhc_pre_split_mixes_fwd_kernel(
        # Input
        input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype],
        mhc_scale: T.Tensor[(3,), dtype],
        mhc_base: T.Tensor[mhc_mult3, dtype],
        # Output
        pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            input_mixes_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)
            pre_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
            post_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
            comb_res_mix_frag = T.alloc_fragment((token_block_size, mhc_mult2), dtype)

            T.annotate_layout(
                {
                    input_mixes_frag: T.Fragment(
                        (token_block_size, mhc_mult3),
                        lambda i, j: (i % 32, i // 32 * mhc_mult3 + j),
                    ),
                },
            )

            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_frag)

            for i, j in T.Parallel(token_block_size, mhc_mult):
                pre_layer_mix_frag[i, j] = (
                    T.sigmoid(
                        input_mixes_frag[i, j] * mhc_scale[0] + mhc_base[j],
                    )
                    + mhc_pre_eps
                )
            for i, j in T.Parallel(token_block_size, mhc_mult):
                post_layer_mix_frag[i, j] = T.sigmoid(input_mixes_frag[i, j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
            for i, j in T.Parallel(token_block_size, mhc_mult2):
                comb_res_mix_frag[i, j] = input_mixes_frag[i, j + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2]

            T.copy(pre_layer_mix_frag, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_frag, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_frag, comb_res_mix[pid * token_block_size, 0])

    return mhc_pre_split_mixes_fwd_kernel
