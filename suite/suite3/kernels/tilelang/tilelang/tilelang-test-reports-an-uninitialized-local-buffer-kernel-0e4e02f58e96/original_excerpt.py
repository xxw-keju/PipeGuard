    @T.prim_func
    def kernel(C: T.Tensor((NB,), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            acc = T.alloc_local((1,), torch.float32)
            C[bx] = acc[0]
