    @T.prim_func
    def main(
        A: T.Tensor((64, 64), T.float16),
        B: T.Tensor((64, 128), T.int8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_stage = T.alloc_shared((2, 64, 64), T.float16, scope="shared.dyn")
            A_i8 = T.view(A_stage, (2, 64, 128), dtype=T.int8)
            T.annotate_layout({A_i8: T.Layout((2, 64, 128), lambda s, i, j: [s, i, j])})

            for i, j in T.Parallel(64, 64):
                A_stage[0, i, j] = A[i, j]

            for i, j in T.Parallel(64, 128):
                B[i, j] = A_i8[0, i, j]
