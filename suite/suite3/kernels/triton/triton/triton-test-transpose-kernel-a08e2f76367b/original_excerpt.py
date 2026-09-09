    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        off2d = off[None, :] + (tl.arange(0, 2) * SIZE)[:, None]
        x = tl.load(X + off2d)
        z = x.T
        tl.store(Z + off2d.T, z)
