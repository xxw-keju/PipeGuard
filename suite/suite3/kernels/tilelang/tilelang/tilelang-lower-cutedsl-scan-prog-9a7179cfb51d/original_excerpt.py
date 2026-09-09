        @T.prim_func
        def prog(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
            with T.Kernel(1, threads=32):
                A_shared = T.alloc_shared((32,), "float32")
                T.copy(A, A_shared)
                T.cumsum(A_shared, dim=0)
                T.copy(A_shared, B)
