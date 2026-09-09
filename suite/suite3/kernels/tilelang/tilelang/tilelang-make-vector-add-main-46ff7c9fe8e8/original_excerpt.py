    @T.prim_func
    def main(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        with T.Kernel(1024):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0  # span_marker_vadd_store
