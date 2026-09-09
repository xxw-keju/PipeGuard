@triton.jit
def cumsum_kernel(ptr):
    block = ptr + tl.arange(0, 4)
    x = tl.load(block)
    tl.store(block, tl.cumsum(x, 0))
