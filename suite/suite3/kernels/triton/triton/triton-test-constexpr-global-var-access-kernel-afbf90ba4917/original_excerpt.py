    @triton.jit
    def kernel():
        a = CONSTEXPR_GLOBAL  # noqa
