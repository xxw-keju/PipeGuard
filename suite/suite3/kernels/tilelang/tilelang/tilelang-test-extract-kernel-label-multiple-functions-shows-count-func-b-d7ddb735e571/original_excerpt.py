    @T.prim_func
    def func_b(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(threads=16):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 2.0
