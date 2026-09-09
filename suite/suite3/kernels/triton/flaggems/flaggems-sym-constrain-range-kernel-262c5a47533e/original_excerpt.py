@triton.jit
def sym_constrain_range_kernel(
    size_ptr,
    min_ptr,
    max_ptr,
    out_ptr,
):
    # Perform the range check on device: load the scalar value and its bounds,
    # then compute a violation flag (1 if out of range, 0 otherwise). This is the
    # actual work of sym_constrain_range -- validating that `size` lies in
    # [min, max] -- expressed as a Triton kernel rather than host-side Python.
    size = tl.load(size_ptr)
    lo = tl.load(min_ptr)
    hi = tl.load(max_ptr)
    below = size < lo
    above = size > hi
    violated = below or above
    tl.store(out_ptr, violated.to(tl.int32))
