@triton.jit
def _pack_4bit_weight_kernel(
    weights,
    scales_zeros,
    bias,
    output,
    weight_elements,
    scale_elements,
    total_elements,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    weight_mask = mask & (offsets < weight_elements)
    values = tl.where(
        weight_mask,
        tl.load(weights + offsets, mask=weight_mask, other=0).to(tl.float32),
        values,
    )

    scale_offsets = offsets - weight_elements
    scale_mask = mask & (scale_offsets >= 0) & (scale_offsets < scale_elements)
    values = tl.where(
        scale_mask,
        tl.load(scales_zeros + scale_offsets, mask=scale_mask, other=0.0).to(
            tl.float32
        ),
        values,
    )

    if HAS_BIAS:
        bias_offsets = scale_offsets - scale_elements
        bias_mask = mask & (bias_offsets >= 0)
        values = tl.where(
            bias_mask,
            tl.load(bias + bias_offsets, mask=bias_mask, other=0.0).to(tl.float32),
            values,
        )

    tl.store(output + offsets, values, mask=mask)
