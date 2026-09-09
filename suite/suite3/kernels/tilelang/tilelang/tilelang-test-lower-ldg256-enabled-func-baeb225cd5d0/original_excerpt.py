    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(8):
                B[i * 8 + j] = A[i * 8 + j]
