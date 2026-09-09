    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        C: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            A_shared = T.alloc_shared((16,), T.float16)
            for i in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
                base: T.int32 = i * 16
                T.copy(A[base], A_shared)
                value: T.float16 = A_shared[0]
                C[base] = value
