    @T.prim_func
    def main(A_ptr: T.ptr):
        A = T.make_tensor(A_ptr, (16,), T.float32)  # span_marker_make_tensor
        with T.Kernel(1, threads=1):
            T.evaluate(A[0])
