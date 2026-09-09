    @triton.jit
    def kernel():
        tl.arange(0, 16)
