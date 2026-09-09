    @tilelang.jit(compile_flags=["-O3", "-DENABLE_BF16"])
    def _splitk_reduce_func(elems_per_cta: int = 1024) -> Callable:
        def _slice_sum(w, gi, gj):
            expr = w[0, gi, gj]
            for s in range(1, split_k):
                expr = expr + w[s, gi, gj]
            return expr

        @T.prim_func
        def _splitk_reduce_main(
            w: T.Tensor((split_k, m, n), accum_dtype),  # type: ignore
            c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            total = m * n
            with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx:
                base = bx * elems_per_cta
                for t in T.Parallel(elems_per_cta):
                    idx = base + t
                    if idx < total:
                        gi = idx // n
                        gj = idx % n
                        c[gi, gj] = T.cast(_slice_sum(w, gi, gj), dtype)

        return _splitk_reduce_main
