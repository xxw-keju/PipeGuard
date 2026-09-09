    @T.prim_func
    def func(B: T.Buffer((128,), "float32"), pred: T.int32):
        S = T.alloc_buffer((128,), dtype=T.float32, scope="shared")
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = S[i * 4 + j]
