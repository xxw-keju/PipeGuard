    @T.prim_func
    def kernel(A: T.Tensor((NB,), torch.int32), C: T.Tensor((NB,), torch.int32)):
        with T.Kernel(NB, threads=64) as bx:
            idx = T.alloc_local((1,), torch.int32)
            idx[0] = 0
            for i in T.serial(NB):
                idx[0] = T.if_then_else(A[i] > 0, i, idx[0])
            C[bx] = idx[0]
