    @T.prim_func
    def kernel(A: T.Tensor((BC, BK), torch.float32), B: T.Tensor((BC,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            a_f = T.alloc_fragment((BC, BK), torch.float32)
            r_f = T.alloc_reducer((BC,), torch.float32, op="sum", replication="all")
            T.clear(r_f)
            T.copy(A, a_f)
            for i, j in T.Parallel(BC, BK):
                r_f[i] += a_f[i, j]
            T.finalize_reducer(r_f)
            T.copy(r_f, B)
