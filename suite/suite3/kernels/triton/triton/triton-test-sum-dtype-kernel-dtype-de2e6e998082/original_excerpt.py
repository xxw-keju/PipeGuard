    @triton.jit
    def kernel_dtype(out_ptr, init, in_dtype: tl.constexpr, out_dtype: tl.constexpr):
        x = tl.full((32, 32), init, dtype=in_dtype)
        x = tl.sum(x, dtype=out_dtype)
        tl.store(out_ptr, x.to(tl.int32))
