@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def avg_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=0.0
            )

            sum_acc += tl.where(in_mask, current_val, 0.0)
            count_acc += in_mask.to(tl.int32)

    if divisor_override != 0:
        divisor = tl.full((BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32)
    elif COUNT_INCLUDE_PAD:
        divisor = tl.full((BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32)
    else:
        divisor = count_acc.to(tl.float32)

    output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(
        output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask
    )
