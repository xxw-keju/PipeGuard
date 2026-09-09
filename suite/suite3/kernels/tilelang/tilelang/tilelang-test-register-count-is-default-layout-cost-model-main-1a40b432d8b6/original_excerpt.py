    @T.prim_func
    def main(
        S: T.Tensor((2,), T.float32),
        Out: T.Tensor((2, 2560), T.float32),
    ):
        with T.Kernel(1, threads=256):
            s_frag = T.alloc_fragment((2,), T.float32)
            for i in T.Parallel(2):
                s_frag[i] = S[i]
            for i, j in T.Parallel(2, 2560):
                Out[i, j] = s_frag[i] * 2.0
