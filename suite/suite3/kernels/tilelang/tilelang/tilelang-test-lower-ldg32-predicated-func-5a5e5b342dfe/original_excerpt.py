    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            B[i] = T.if_then_else(pred > 0, A[i], T.float32(0))
