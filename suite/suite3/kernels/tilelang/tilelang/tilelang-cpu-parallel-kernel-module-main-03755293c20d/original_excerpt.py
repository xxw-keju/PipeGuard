    @T.prim_func
    def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((16,), "int32")
            for i in T.Parallel(16):
                frag[i] = A[i]
            for i in T.Parallel(16):
                B[i] = frag[i]
