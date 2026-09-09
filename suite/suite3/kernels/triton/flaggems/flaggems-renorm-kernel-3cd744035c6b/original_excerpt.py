@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("renorm_"),
    key=["N"],
)
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_kernel(X, N, p, maxnorm, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x = tl.load(X + pid * N + offset, mask=mask, other=0.0).to(tl.float32)
    _sum = tl.sum(tl_extra_shim.pow(tl.abs(x), p))

    norm = tl_extra_shim.pow(_sum, 1.0 / p)
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)

    x = x * scale
    tl.store(X + pid * N + offset, x, mask=mask)
