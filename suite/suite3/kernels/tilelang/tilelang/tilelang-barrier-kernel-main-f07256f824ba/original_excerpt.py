    @T.prim_func
    def main(C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1) as _:
            bar = T.alloc_barrier(1)  # noqa: F841
            C[0] = 0.0
