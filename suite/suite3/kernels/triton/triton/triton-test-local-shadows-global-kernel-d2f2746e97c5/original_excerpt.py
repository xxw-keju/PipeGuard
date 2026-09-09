    @triton.jit
    def kernel():
        _, GLOBAL = 0, 0  # noqa
        a = GLOBAL  # noqa
