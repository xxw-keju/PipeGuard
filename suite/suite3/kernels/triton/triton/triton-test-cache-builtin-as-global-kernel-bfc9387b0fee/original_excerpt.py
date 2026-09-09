    @triton.jit
    def kernel():
        x = BUILTIN_AS_GLOBAL  # noqa
