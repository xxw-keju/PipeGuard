    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(128, "threadIdx.x"):
            B[i] = A[i]
