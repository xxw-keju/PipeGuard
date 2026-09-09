    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
            base: T.int32 = i * 16
            T.copy(A[base], S)
            value: T.float16 = S[0]
            C[base] = value
