    @triton.jit
    def kernel(out):
        tl.store(out, 1)
