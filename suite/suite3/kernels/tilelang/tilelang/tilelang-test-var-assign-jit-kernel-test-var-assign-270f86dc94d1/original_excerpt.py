        @T.prim_func
        def test_var_assign(A: T.Tensor((2,), T.int32)):
            with T.Kernel(1) as _:
                a = T.alloc_var(T.int32, init=1)
                b = T.alloc_var(T.int32, init=a)  # b gets value of a
                a = 2
                d = T.alloc_var(T.int32, init=a)  # c gets new value of a
                A[0] = b
                A[1] = d
