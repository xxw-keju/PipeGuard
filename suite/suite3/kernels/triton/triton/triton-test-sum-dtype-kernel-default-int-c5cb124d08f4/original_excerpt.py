    @triton.jit
    def kernel_default_int(out_ptr):
        x = tl.full((32, 32), 1, dtype=tl.int1)
        x = tl.sum(x)
        tl.store(out_ptr, x)
