    @T.prim_func
    def kernel(
        S: T.Tensor((BC,), torch.float32, scope="shared.dyn"),
        C: T.Tensor((BC,), torch.float32),
    ):
        with T.Kernel(1, threads=64) as _:
            for i in T.Parallel(BC):
                C[i] = S[i]
