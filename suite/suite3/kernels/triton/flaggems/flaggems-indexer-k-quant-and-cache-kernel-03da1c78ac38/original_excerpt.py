@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_mapping_ptr,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    quant_block_id = tl.program_id(1) * 4
    quant_block_offsets = tl.arange(0, 4)
    head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
    offsets = (
        quant_block_id + quant_block_offsets[:, None]
    ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
    mask = offsets < head_dim

    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return

    block_id = slot_id // block_size
    block_offset = slot_id % block_size

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    amax = tl.max(tl.abs(val).to(tl.float32), axis=1)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale[:, None]).to(kv_cache_ptr.type.element_ty)
    dst_ptr = kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
    tl.store(dst_ptr + offsets, fp8_val, mask=mask)

    dst_scale_ptr = (
        kv_cache_scale_ptr
        + block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    scale_mask = quant_block_id + quant_block_offsets < num_quant_blocks
    tl.store(dst_scale_ptr + quant_block_offsets, scale, mask=scale_mask)
