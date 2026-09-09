@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("fractional_max_pool2d_backward"),
    key=["out_h", "out_w"],
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def fractional_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Output strides (element-based, contiguous)
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Backward kernel for fractional max pool 2d.

    Scatters gradients back to input positions using the indices from forward.
    Uses atomic_add since multiple output positions may map to the same input.
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    h_mask = h_out_offsets < out_h
    w_mask = w_out_offsets < out_w
    out_mask = h_mask[:, None] & w_mask[None, :]

    # Load grad_output and indices for this tile
    out_base = pid_nc * out_stride_nc
    out_offsets = (
        out_base
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
    idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)

    # Scatter gradients to input using indices
    in_nc_offset = pid_nc * in_h * in_w
    in_offsets = in_nc_offset + idx_vals

    valid_mask = out_mask & (idx_vals >= 0)
    tl.atomic_add(
        grad_input_ptr + in_offsets, grad_vals.to(tl.float32), mask=valid_mask
    )
