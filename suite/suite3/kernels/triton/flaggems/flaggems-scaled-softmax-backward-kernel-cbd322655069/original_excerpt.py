@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    query_mask = query_offsets < query_seq_len

    output_row_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_output_row_ptr = (
        grad_output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_input_row_ptr = (
        grad_input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    D = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, cache_modifier=".ca")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, cache_modifier=".ca")

        dot_block = P_block * dP_block
        D += tl.sum(tl.where(mask, dot_block, 0.0), axis=1)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dS = grad_input_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, eviction_policy="evict_first")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, eviction_policy="evict_first")

        dZ_block = P_block * (dP_block - D[:, None])
        dS_block = scale_factor * dZ_block

        tl.store(ptr_dS, dS_block, mask=mask, cache_modifier=".cs")
