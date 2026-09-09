    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        with T.Kernel(1, threads=1):
            result = T.alloc_shared((4,), T.uint32)
            A[0] = T.clc_is_canceled(result)
            A[1] = T.Cast("int32", T.clc_get_first_ctaid_x(result))
