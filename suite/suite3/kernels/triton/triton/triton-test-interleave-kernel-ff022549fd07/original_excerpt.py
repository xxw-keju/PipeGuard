    @triton.jit(debug=debug)
    def kernel(Z, N: tl.constexpr):
        z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
        tl.store(Z + tl.arange(0, 2 * N), z)
