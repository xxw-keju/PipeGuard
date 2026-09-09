    @T.prim_func
    def before(
        KV: T.Tensor((4, 4), T.bfloat16),
        ids: T.Tensor((4,), T.int32),
        C: T.Tensor((4,), T.bfloat16),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared((4,), T.bfloat16)

            for i in T.Pipelined(4, num_stages=2):
                if i > 1:
                    idx = ids[i]
                    T.copy(KV[idx, :], A)
                    T.copy(A, C)
