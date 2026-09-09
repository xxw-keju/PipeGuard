    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), N: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                idx = i * 4 + j
                B[idx] = T.if_then_else(idx < N, A[idx], T.float32(0))
