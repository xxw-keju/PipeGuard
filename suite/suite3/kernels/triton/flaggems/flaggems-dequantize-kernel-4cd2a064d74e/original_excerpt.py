@triton.jit
def dequantize_kernel(
    x, scale, zero_point, output, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_val = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    result = (x_val - zero_point) * scale
    tl.store(output + offsets, result, mask=mask)
