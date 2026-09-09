    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)
