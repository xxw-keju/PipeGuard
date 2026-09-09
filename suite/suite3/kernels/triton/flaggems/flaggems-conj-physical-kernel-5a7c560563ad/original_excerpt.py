@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conj_physical"),
    key=["n_elements"],
)
@triton.jit
def conj_physical_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    base = offsets * 2
    real = tl.load(in_ptr + base, mask=mask)
    imag = tl.load(in_ptr + base + 1, mask=mask)

    tl.store(out_ptr + base, real, mask=mask)
    tl.store(out_ptr + base + 1, -imag, mask=mask)
