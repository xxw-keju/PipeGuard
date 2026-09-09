    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            B_shared = T.alloc_shared((64,), "float32")
            T.copy(A[0:128], B_shared[0:64])  # span_marker_bad_copy
