    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            f = T.alloc_fragment((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = f[i]
            T.clear(f)
