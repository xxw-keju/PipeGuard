    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                B[i * 4 + j] = T.if_then_else(pred > 0, A[i * 4 + j], T.float32(0))
