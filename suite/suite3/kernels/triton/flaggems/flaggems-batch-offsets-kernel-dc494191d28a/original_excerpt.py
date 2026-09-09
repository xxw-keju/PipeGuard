@libentry()
@triton.jit
def _batch_offsets_kernel(
    batch_sizes_ptr,
    offsets_ptr,
    bs32_ptr,
    num_steps,
    BLOCK: tl.constexpr,
):
    # Exclusive prefix-sum of batch_sizes (plus an int32 copy for the recurrence mask),
    # in one program via tl.cumsum. Avoids torch.cumsum/sub dispatch under use_gems().
    offs = tl.arange(0, BLOCK)
    mask = offs < num_steps
    vals = tl.load(batch_sizes_ptr + offs, mask=mask, other=0).to(tl.int32)
    exclusive = tl.cumsum(vals, axis=0) - vals
    tl.store(offsets_ptr + offs, exclusive, mask=mask)
    tl.store(bs32_ptr + offs, vals, mask=mask)
