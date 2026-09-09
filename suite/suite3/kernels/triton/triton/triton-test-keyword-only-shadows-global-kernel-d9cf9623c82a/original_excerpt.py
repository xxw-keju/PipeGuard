    @triton.jit
    def kernel(*, GLOBAL: tl.constexpr):
        tl.static_assert(GLOBAL == 1)
