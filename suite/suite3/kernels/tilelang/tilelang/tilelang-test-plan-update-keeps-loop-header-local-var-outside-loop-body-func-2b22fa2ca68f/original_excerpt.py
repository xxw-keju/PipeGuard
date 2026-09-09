    @T.prim_func
    def func(x: T.Tensor((256,), "int64")):
        with T.Kernel(256, threads=128):
            a, b = T.alloc_var(T.int), T.alloc_var(T.int)
            T.fill(x[a:b], 0)
