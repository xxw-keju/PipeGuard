    @T.prim_func
    def kernel(
        source: T.Tensor((4,), "int4"),
        output: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                output[i] = T.cast(source[i], "int32")
