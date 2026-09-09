    @T.prim_func
    def kernel(A: T.Tensor((4, 32), T.float8_e4m3fn), B: T.Tensor((4,), T.float16)):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((4, 32), T.float8_e4m3fn)
            dst = T.alloc_fragment((4,), T.float16)
            T.copy(A, src)
            T.reduce_absmax(src, dst, dim=1)
            T.copy(dst, B)
