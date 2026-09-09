    @T.prim_func
    def prog(A: T.Tensor((1,), "int8"), B: T.Tensor((1,), "uint16")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "int8")
            local[0] = A[0]
            B[0] = T.cast(local[0] >> 1, T.uint16)
