    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = A[i * 4 + j]
