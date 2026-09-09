    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:  # span_marker_kernel
            A_shared = T.alloc_shared((128, 128), "float16")  # span_marker_alloc_shared
            C_local = T.alloc_fragment((128, 128), "float32")  # span_marker_alloc_fragment
            T.copy(A, A_shared)  # span_marker_copy
            T.clear(C_local)  # span_marker_clear
            for _k in T.serial(8):  # span_marker_for
                T.gemm(A_shared, A_shared, C_local)  # span_marker_gemm
            if bx == 0:  # span_marker_if
                T.copy(C_local, C)  # span_marker_copy_out
