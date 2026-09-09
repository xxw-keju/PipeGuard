    @T.prim_func
    def kernel(out: T.Tensor[(4,), T.float32]):
        with T.Kernel(0, threads=1) as pid:
            out[pid] = T.float32(1.0)
