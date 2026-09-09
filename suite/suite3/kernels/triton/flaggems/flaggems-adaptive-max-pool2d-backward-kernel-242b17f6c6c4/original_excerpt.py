@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("adaptive_max_pool2d_backward"),
    key=["out_numel"],
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def adaptive_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_h,
    in_w,
    out_h,
    out_w,
    out_numel,
    in_numel_per_nc,
    # Strides for grad_output/indices (contiguous, so (N*C, out_h*out_w))
    out_stride_nc,
    # Tiling
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a block of output elements.
    For each output element, scatter its grad_output value to the position
    indicated by the corresponding index in grad_input.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_numel

    # Load grad_output and indices
    grad_out_val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    indices_val = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Calculate which (n*c) batch this output element belongs to
    nc_idx = offsets // out_stride_nc

    # Calculate the position in grad_input
    # indices_val is the flat index within (in_h * in_w) spatial plane
    grad_input_offset = nc_idx * in_numel_per_nc + indices_val

    # Use atomic add since multiple output elements could map to the same input
    # (though for adaptive_max_pool2d this shouldn't happen, but using atomic
    # for correctness)
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_out_val, mask=mask)
