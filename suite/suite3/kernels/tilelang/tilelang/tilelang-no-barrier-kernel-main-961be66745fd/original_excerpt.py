    @T.prim_func
    def main(C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1) as _:
            C[0] = 0.0
