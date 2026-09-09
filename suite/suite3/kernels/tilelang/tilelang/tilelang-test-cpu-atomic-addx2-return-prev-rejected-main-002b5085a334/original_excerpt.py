    @T.prim_func
    def main(
        Dst: T.Tensor((2,), "float16"),
        Val: T.Tensor((2,), "float16"),
        Prev: T.Tensor((2,), "float16"),
    ):
        with T.Kernel(1):
            Prev[0:2] = T.atomic_addx2(Dst[0:2], Val[0:2], return_prev=True)
