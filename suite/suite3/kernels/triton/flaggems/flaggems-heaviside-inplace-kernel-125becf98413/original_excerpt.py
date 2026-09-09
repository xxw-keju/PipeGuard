@triton.jit
def _heaviside_inplace_kernel(x_ptr, v_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    is_zero = x == 0
    is_neg = x < 0
    is_pos = x > 0

    # heaviside(x, v) = 0 if x < 0, 1 if x > 0, v if x == 0.
    # For NaN inputs none of the comparisons hold; x + x propagates NaN.
    res = tl.where(is_zero, v, tl.where(is_neg, 0, tl.where(is_pos, 1, x + x)))
    tl.store(x_ptr + offsets, res, mask=mask)
