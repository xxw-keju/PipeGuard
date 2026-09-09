    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, order=[1, 0], stage=[0, 1]):
            base: T.int32 = i * 16
            T.copy(A[base], S)
            T.copy(S, C[base])
