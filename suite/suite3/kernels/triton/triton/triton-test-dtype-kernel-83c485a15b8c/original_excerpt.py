    @triton.jit
    def kernel(X):
        dtype_x: tl.constexpr = X.dtype.element_ty
        tl.static_assert(dtype_x == tl.int32)
        tl.static_assert(dtype_x == tl.constexpr(tl.int32))
        tl.static_assert(dtype_x == tl.int8 or (dtype_x == tl.int16 or dtype_x == tl.int32))
