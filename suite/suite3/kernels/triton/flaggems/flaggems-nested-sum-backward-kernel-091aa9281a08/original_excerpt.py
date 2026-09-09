@triton.jit
def _nested_sum_backward_kernel(
    grad_ptr,
    out_ptr,
    numel,
    dim_size,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decompose the flattened output index into (outer, dim, inner) where the
    # reduced `dim` has been broadcast from `grad`, which stores a single value
    # per (outer, inner) pair. `grad` is laid out as [outer, inner] in memory.
    outer_idx = offsets // (dim_size * inner_size)
    inner_idx = offsets % inner_size
    grad_offset = outer_idx * inner_size + inner_idx

    grad = tl.load(grad_ptr + grad_offset, mask=mask)
    tl.store(out_ptr + offsets, grad, mask=mask)
