    @T.prim_func
    def after(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[1023, 510, 253]  # -1+1024=1023, -2+512=510, -3+256=253
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
