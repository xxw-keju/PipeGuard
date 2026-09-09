@libentry()
@triton.jit
def histc_kernel_simple(
    inp_ptr,
    out_ptr,
    n_elements,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple histogram kernel - each program handles one element at a time.
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=float("nan"))

    # Convert to float32 for computation
    inp_val = inp_val.to(tl.float32)

    # Compute bin indices using multiplication to avoid float division precision loss
    bin_idx = tl.floor((inp_val - min_val) * bins / (max_val - min_val)).to(tl.int64)

    # Handle edge case: elements exactly equal to max go to last bin
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)

    # Check if elements are in valid range (excludes NaN)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)

    # Clamp bin indices to valid range
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

    valid_mask = mask & in_range

    # Atomically add to histogram
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem="relaxed")
