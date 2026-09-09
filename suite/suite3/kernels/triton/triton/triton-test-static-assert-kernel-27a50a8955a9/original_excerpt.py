    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.static_assert(COND)
