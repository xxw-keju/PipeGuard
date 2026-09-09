        @T.prim_func
        def prim_buffer_slice_step(A: T.Buffer((10,), T.int32), B: T.Buffer((5,), T.int32)):
            with T.Kernel(1):
                B[0:5:2] = A[0:10:2]
