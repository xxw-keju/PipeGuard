    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(64, "threadIdx.x"):
            for j in T.vectorized(2):
                B[i * 2 + j] = A[i * 2 + j]
