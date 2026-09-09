    @triton.jit
    def dtype_tensor_kernel(dtype: tl.constexpr):
        tensor = tl.zeros((1, ), dtype)
