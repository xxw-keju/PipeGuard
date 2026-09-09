@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_forward_kernel(
    output_ptr,
    input_ptr,
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

    row_start_ptr = (
        input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    m = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    exp_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        block_ptr = row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr, mask=mask, other=-float("inf"), cache_modifier=".ca"
        )
        s_block = s_block * scale_factor

        m_new = tl.max(s_block, axis=1)
        m_old = m
        m = tl.maximum(m_old, m_new)

        s_prev = tl.exp(m_old - m)
        exp_sum = exp_sum * s_prev

        s_curr = tl.exp(s_block - m[:, None])
        l_new = tl.sum(tl.where(mask, s_curr, 0.0), axis=1)
        exp_sum = exp_sum + l_new

    exp_sum_inv = 1.0 / exp_sum

    out_row_start_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)

        block_ptr_in = row_start_ptr[:, None] + k_offsets[None, :]
        block_ptr_out = out_row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr_in, mask=mask, other=-float("inf"), eviction_policy="evict_first"
        )

        s_block = s_block * scale_factor
        s_block = s_block - m[:, None]
        p_block = tl.exp(s_block)
        p_block = p_block * exp_sum_inv[:, None]

        tl.store(block_ptr_out, p_block, mask=mask, cache_modifier=".cs")
