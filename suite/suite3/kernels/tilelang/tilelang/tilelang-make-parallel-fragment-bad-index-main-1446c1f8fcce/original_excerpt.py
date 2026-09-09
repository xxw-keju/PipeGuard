    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((4,), "float32")  # span_marker_frag_alloc
            for i in T.Parallel(4):
                frag[1] = A[i]  # span_marker_frag_store
