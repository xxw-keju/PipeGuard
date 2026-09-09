    @triton.jit
    def kernel(Out):
        zero = 0
        tl.store(Out, return_poison(zero))
