@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)
