    @triton.jit
    def kernel():
        tl.static_assert(passthrough_constexpr(TrivialTuple(0)).foo == 0)
