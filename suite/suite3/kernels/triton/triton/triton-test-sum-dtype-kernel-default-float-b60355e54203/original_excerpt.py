    @triton.jit
    def kernel_default_float(out_ptr):
        x = tl.full((32, 32), 1.0, dtype=tl.bfloat16)
        x = tl.sum(x)
        tl.store(out_ptr, x)
