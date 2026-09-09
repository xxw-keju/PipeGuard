@triton.jit
def _hypot_inplace_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute in fp32 for stability, then cast back to the input dtype.
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    out32 = tl.sqrt(x32 * x32 + y32 * y32)

    tl.store(out_ptr + offsets, out32.to(x.dtype), mask=mask)
