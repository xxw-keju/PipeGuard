    @T.prim_func
    def kernel(
        A: T.Tensor((128, 64), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=256):
            src = T.alloc_shared((128, 64), T.float32)
            dst = T.alloc_fragment((128,), T.float32)
            T.copy(A, src, disable_tma=True)
            T.reduce_sum(src, dst, dim=1, batch=4)
            T.copy(dst, B)
