@libentry()
@triton.jit(do_not_specialize=["scale", "zero_point", "quant_min", "quant_max"])
def fake_quantize_per_tensor_affine_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = scale.to(tl.float32)
    inv_scale = inv_scale.to(tl.float32)
    quantized = tl_extra_shim.nearbyint(value * inv_scale) + zero_point
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point) * scale
    tl.store(output_ptr + offsets, output, mask=mask)
