    @tilelang.jit(out_idx=[3])
    def _apply(splits, threads, per):
        chunk = N // splits

        @T.prim_func
        def main(
            summed: T.Tensor([N], dtype),
            partial: T.Tensor([splits], accum),
            weight: T.Tensor([N], dtype),
            y: T.Tensor([N], dtype),
        ):
            with T.Kernel(splits, threads=threads) as bx:
                tx = T.get_thread_binding()
                total = T.alloc_local([1], accum)
                total[0] = T.cast(0, accum)
                for s in T.serial(splits):
                    total[0] += partial[s]
                rrms = T.rsqrt(total[0] / T.cast(N, accum) + T.cast(eps, accum))
                v = T.alloc_local([per], dtype)
                o = T.alloc_local([per], dtype)
                for step in T.serial(chunk // (threads * per)):
                    base = bx * chunk + (step * threads + tx) * per
                    for i in T.vectorized(per):
                        v[i] = summed[base + i]
                    for i in T.serial(per):
                        o[i] = T.cast(
                            T.cast(v[i], accum) * rrms * T.cast(weight[base + i], accum),
                            dtype,
                        )
                    for i in T.vectorized(per):
                        y[base + i] = o[i]

        return main
