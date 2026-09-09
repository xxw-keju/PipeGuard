@libentry()
@triton.jit
def conj_physical__kernel(ptr, n_real_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_real_elements
    x = tl.load(ptr + offsets, mask=mask)
    # even offsets are real parts (kept), odd offsets are imag parts (negated)
    y = tl.where((offsets % 2) == 1, -x, x)
    tl.store(ptr + offsets, y, mask=mask)
