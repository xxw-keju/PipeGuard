    @T.prim_func
    def atomic_or_kernel(A: T.Tensor((1,), T.int32), mask: T.int32):
        with T.Kernel(1, threads=32):
            T.atomic_or(A[0], mask, memory_order="release")
