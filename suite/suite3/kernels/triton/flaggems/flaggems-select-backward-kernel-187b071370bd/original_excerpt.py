@triton.jit
def _select_backward_kernel(
    grad_ptr,
    out_ptr,
    total: tl.constexpr,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    outer = offs // inner_size
    inner = offs % inner_size

    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = outer * dim_stride + index * inner_size + inner

    tl.store(out_ptr + out_offset, vals, mask=mask)
