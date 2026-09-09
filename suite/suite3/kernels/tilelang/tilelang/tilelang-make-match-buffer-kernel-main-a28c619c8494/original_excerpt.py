    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16,), T.float32)  # span_marker_match_buffer
        with T.Kernel(1, threads=1):
            T.evaluate(A[0])
