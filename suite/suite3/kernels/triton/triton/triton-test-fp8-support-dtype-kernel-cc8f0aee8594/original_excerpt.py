    @triton.jit
    def dtype_kernel(dtype: tl.constexpr):
        a = tl.full((64, 64), 0.0, dtype)
        tl.dot(a, a)
