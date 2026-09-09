    @T.prim_func
    def before(
        KV: T.Tensor((4, 4), T.float16),
        ids: T.Tensor((4,), T.int32),
        C: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            A = T.alloc_shared((4,), T.float16)
            for i in T.Pipelined(4, num_stages=2):
                _id = ids[i]
                T.copy(KV[_id, :], A)
                T.copy(A, C)
