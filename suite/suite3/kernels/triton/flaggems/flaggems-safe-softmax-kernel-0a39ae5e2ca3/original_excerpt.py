@triton.jit
def _safe_softmax_kernel(
    input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_offset = row_id * n_cols
    x = tl.load(input_ptr + row_offset + cols, mask=mask, other=-float("inf"))
    x_fp32 = x.to(tl.float32)

    x_max = tl.max(x_fp32, axis=0)
    all_neginf = x_max == -float("inf")

    x_shifted = x_fp32 - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    softmax = tl.where(all_neginf, tl.zeros([BLOCK_SIZE], dtype=tl.float32), softmax)

    tl.store(output_ptr + row_offset + cols, softmax, mask=mask)
