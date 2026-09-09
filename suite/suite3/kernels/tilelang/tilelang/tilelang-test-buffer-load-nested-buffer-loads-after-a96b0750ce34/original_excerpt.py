    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        inner_val = A[1023, 10]
        outer_val = A[inner_val.astype(T.int32), 510]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = outer_val
