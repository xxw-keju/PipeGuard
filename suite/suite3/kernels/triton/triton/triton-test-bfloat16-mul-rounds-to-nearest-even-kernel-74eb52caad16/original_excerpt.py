    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        z = tl.load(X + off) * tl.load(Y + off)
        tl.store(Z + off, z)
