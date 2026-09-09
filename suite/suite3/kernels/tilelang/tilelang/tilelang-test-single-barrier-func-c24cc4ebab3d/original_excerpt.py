    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            mbar = T.alloc_barrier(128)  # noqa: F841
