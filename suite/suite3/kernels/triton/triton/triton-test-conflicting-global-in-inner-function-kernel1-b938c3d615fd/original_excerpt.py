    @triton.jit
    def kernel1():
        a = CONFLICTING_GLOBAL  # noqa
        conflicting_global_inner()
