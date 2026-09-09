    @T.prim_func
    def kernel(
        A: T.Tensor((1,), T.float32),
        B: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4,), T.float32)
            for i in T.serial(4):
                A_shared[i] = T.float32(i + 1)
            T.ptx_cp_async(
                T.access_ptr(A_shared[0], "w", 1),
                T.access_ptr(A[0], "r", 1),
                1,
                predicate=False,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            for i in T.serial(4):
                B[i] = A_shared[i]
