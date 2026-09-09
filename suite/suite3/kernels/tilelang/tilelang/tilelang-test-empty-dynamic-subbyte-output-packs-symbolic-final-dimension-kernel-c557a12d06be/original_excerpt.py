    @T.prim_func
    def kernel(n: T.int32):
        output = T.empty((3, n * 2), T.float4_e2m1fn)
        with T.Kernel(1, threads=1):
            T.evaluate(0)
        return output
