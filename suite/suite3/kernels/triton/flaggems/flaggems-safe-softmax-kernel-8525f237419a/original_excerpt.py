@triton.jit
def _safe_softmax_kernel(
    input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = input_ptr + row_id * n_cols
    out_start = output_ptr + row_id * n_cols

    m_i = -float("inf")
    l_i = 0.0

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        m_ij = tl.max(x_fp32, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_i_new))
        beta = tl.exp(x_fp32 - m_i_new)

        sum_block = tl.sum(tl.where(mask, beta, 0.0), axis=0)
        l_i = l_i * alpha + sum_block
        m_i = m_i_new

    all_neginf = m_i == -float("inf")

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        p = tl.exp(x_fp32 - m_i) / l_i
        p = tl.where(all_neginf, 0.0, p)
        tl.store(out_start + cols, p, mask=mask)
