    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), "float16")
            B_shared = T.alloc_shared((128, 128), "float16")
            for _k in T.Pipelined(2, order=[0, 0], stage=[0, 1]):  # span_marker_pipelined
                T.copy(A, A_shared)  # span_marker_pipe_copy1
                T.copy(B, B_shared)  # span_marker_pipe_copy2
