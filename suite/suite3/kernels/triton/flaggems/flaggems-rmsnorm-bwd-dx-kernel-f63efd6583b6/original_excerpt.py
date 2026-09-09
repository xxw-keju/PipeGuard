@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rmsnorm_bwd_dx"),
    key=["M", "N"],
)
@triton.jit
def rmsnorm_bwd_dx_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    M,
    N,
    zero_centered_gamma: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute dx for RMSNorm backward using 2D tiling.
    Each program handles BLOCK_ROW_SIZE rows.
    """
    pid = ext.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M

    # Determine compute dtype
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    # Setup pointers with row offsets
    dz_ptr = dz_ptr + pid * N
    x_ptr = x_ptr + pid * N
    dx_ptr = dx_ptr + pid * N

    # Load rsigma for each row
    rsigma = tl.load(rsigma_ptr + pid, mask=row_mask).to(cdtype)

    # First pass: compute c1 = mean(x_hat * dz * w) for each row
    c1_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        c1_acc += tl.where(mask, x_hat * dz * w, 0.0)

    c1 = tl.sum(c1_acc, axis=1)[:, None] / N

    # Second pass: compute dx = rsigma * (dz * w - x_hat * c1)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        dx = rsigma * (dz * w - x_hat * c1)

        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
