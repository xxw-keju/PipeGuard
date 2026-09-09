    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_small_batch_func(
        block_n: int = 1,
        reduce_threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        tile_k = 128 // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemm_small_batch_main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor((n, k), dtype),
            c: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((m, tile_k), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    for mi in T.serial(m):
                        for _k in T.vectorized(tile_k):
                            a_local[mi, _k] = a[mi, bk * block_k + tk * tile_k + _k]
                    for mi in T.serial(m):
                        for _k in T.serial(tile_k):
                            c_accum[mi] += a_local[mi, _k].astype(accum_dtype) * b_shared[
                                tn, tk * tile_k + _k
                            ].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                for mi in T.serial(m):
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1), c_accum[mi], True, c_reduced[0], tk, dtype="handle"
                            )
                        )
                    if bn * block_n + tn < n:
                        c[mi, bn * block_n + tn] = c_reduced[0]

        return _gemm_small_batch_main
