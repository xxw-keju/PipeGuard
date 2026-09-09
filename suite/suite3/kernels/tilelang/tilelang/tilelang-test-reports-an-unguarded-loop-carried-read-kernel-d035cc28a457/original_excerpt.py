    @T.prim_func
    def kernel(A: T.Tensor((BC,), torch.float32), C: T.Tensor((BC,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            f = T.alloc_fragment((1,), torch.float32)
            for k in T.serial(BC):
                C[k] = f[0]
                f[0] = A[k]
