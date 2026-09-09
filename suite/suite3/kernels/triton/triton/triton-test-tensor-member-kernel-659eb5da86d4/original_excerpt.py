    @triton.jit
    def kernel():
        x = tl.arange(0, 16)
        tl.device_assert(tl.abs(x) == x.abs())
        tl.device_assert(tl.sum(x) == x.sum())
