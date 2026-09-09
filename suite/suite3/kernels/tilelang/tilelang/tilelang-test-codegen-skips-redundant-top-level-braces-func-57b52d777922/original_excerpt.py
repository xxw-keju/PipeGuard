    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            C = T.alloc_fragment((128,), T.float32)
            T.clear(C)
            for k in T.serial(4):
                with T.sblock():
                    T.sblock_attr({"lexical_alloc_scope": 1})
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    C[T.get_thread_binding()] = S[T.get_thread_binding()]
                # Sibling statement keeps the inner scope's braces from being
                # elided as redundant with the loop body.
                C[T.get_thread_binding()] = C[T.get_thread_binding()] + T.float32(1)
            for k in T.serial(4):
                B[T.get_thread_binding(), k] = C[T.get_thread_binding()]
