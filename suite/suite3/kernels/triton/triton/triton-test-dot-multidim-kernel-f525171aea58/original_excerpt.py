    @triton.jit
    def kernel(X, Y, Z, RANK: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        x = tl.load(X + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
        y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))
