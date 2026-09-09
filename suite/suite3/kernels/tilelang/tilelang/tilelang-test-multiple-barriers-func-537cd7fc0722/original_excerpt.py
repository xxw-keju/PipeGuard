    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            mbars = T.alloc_barrier([1, 1, 128, 128])  # noqa: F841
