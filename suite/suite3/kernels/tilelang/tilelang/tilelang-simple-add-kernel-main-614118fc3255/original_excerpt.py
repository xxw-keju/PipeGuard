    @T.prim_func
    def main(
        x: T.Tensor((128,), T.float32),
        y: T.Tensor((128,), T.float32),
    ):
        # One-dimensional kernel; writes y from x without modifying x
        with T.Kernel(128, threads=32) as pid:
            y[pid] = x[pid] + 1.0
