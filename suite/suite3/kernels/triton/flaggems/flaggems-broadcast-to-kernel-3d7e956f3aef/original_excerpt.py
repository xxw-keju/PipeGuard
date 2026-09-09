@triton.jit
def broadcast_to_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    ndims,
    out_shape_ptr,
    out_cumprod_ptr,
    in_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute input offsets corresponding to each output linear index
    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Accumulate contributions per dimension
    for d in range(MAX_DIMS):
        # Load scalars defining the output decomposition and input strides
        s = tl.load(out_shape_ptr + d)
        stride_right = tl.load(out_cumprod_ptr + d)
        in_stride = tl.load(in_stride_ptr + d)
        # idx along dimension d for each linear offset
        idx_d = (offsets // stride_right) % s
        # contribution to input linear offset
        in_offsets += idx_d * in_stride

    # Load from input using computed offsets and store to output
    x = tl.load(x_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)
