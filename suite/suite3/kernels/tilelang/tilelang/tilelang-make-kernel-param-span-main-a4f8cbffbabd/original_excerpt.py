    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),  # span_marker_param_a
        B: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            T.copy(A, B)
