        @T.prim_func
        def kernel(out: T.Tensor((1,), T.float32)) -> None:
            with T.Kernel(1, threads=32) as _:
                v = T.alloc_var(T.float32, init=3.14)
                if T.get_thread_binding() == 0:
                    out[0] = v
