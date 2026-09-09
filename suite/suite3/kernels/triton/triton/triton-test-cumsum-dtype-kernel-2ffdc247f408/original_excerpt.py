    @triton.jit
    def kernel(Z):
        x = tl.full((4, ), True, dtype=tl.int1)
        z = tl.cumsum(x, axis=0)
        tl.store(Z + tl.arange(0, 4), z)
