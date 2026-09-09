    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            for k in T.serial(4):
                with T.sblock():
                    T.sblock_attr({"lexical_alloc_scope": 1})
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    B[T.get_thread_binding(), k] = S[T.get_thread_binding()]
