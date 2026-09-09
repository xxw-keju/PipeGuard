        @tilelang.jit(out_idx=[1])
        def _func(threads: int):
            step = threads * vec
            steps = n // step
            exact = steps * step == n

            @T.prim_func
            def main(x: T.Tensor((m, n), dtype), out: T.Tensor((m,), out_dtype)):
                with T.Kernel(m, threads=threads) as bm:
                    acc = T.alloc_fragment((threads, vec), "float32")
                    lane = T.alloc_fragment((threads,), "float32")
                    red = T.alloc_fragment((1,), "float32")
                    T.clear(acc)
                    for k in T.serial(steps):
                        for i, j in T.Parallel(threads, vec):
                            acc[i, j] += T.Cast("float32", x[bm, (k * threads + i) * vec + j])
                    if not exact:
                        for i, j in T.Parallel(threads, vec):
                            col = steps * step + i * vec + j
                            with T.If(col < n):  # noqa: SIM117
                                with T.Then():
                                    acc[i, j] += T.Cast("float32", x[bm, col])
                    T.reduce_sum(acc, lane, dim=1)
                    T.reduce_sum(lane, red, dim=0)
                    total = red[0] / float(n) if op_kind == "mean" else red[0]
                    out[bm] = T.Cast(out_dtype, total)

            return main
