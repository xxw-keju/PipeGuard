        @tilelang.jit(out_idx=[1])
        def _func():
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(M, threads=threads) as row:
                    tx = T.get_thread_binding()
                    running = T.alloc_local((1,), "float32")
                    warp_prod = T.alloc_shared((num_warps,), "float32")
                    held = T.alloc_local((_PROD_POLICY.cols_per_thread,), dtype)
                    staged = T.alloc_shared((1 if vectorized else chunk,), dtype)
                    # One independent fp32 chain per slot; a single running product
                    # would serialize every multiply behind the previous one.
                    slots = T.alloc_local((_PROD_POLICY.cols_per_thread,), "float32")

                    for c in T.serial(_PROD_POLICY.cols_per_thread):
                        slots[c] = T.cast(1.0, "float32")
                    if vectorized:
                        for t in T.serial(full_tiles):
                            for g in T.serial(vec_groups):
                                for c in T.vectorized(vec_elems):
                                    held[g * vec_elems + c] = x[
                                        row,
                                        t * chunk
                                        + tx * _PROD_POLICY.cols_per_thread
                                        + g * vec_elems
                                        + c,
                                    ]
                            for c in T.serial(_PROD_POLICY.cols_per_thread):
                                slots[c] = slots[c] * T.cast(held[c], "float32")
                    else:
                        for t in T.serial(tiles):
                            for i in T.Parallel(chunk):
                                staged[i] = x[row, t * chunk + i]
                            T.sync_threads()
                            for c in T.serial(_PROD_POLICY.cols_per_thread):
                                kept = staged[tx * _PROD_POLICY.cols_per_thread + c]
                                if exact:
                                    slots[c] = slots[c] * T.cast(kept, "float32")
                                else:
                                    col = (t * threads + tx) * _PROD_POLICY.cols_per_thread + c
                                    slots[c] = slots[c] * T.if_then_else(
                                        col < N, T.cast(kept, "float32"), T.cast(1.0, "float32")
                                    )
                            T.sync_threads()

                    running[0] = slots[0]
                    for c in T.serial(1, _PROD_POLICY.cols_per_thread):
                        running[0] = running[0] * slots[c]

                    for stage in T.serial(_WARP_STAGES):
                        running[0] = running[0] * T.shfl_xor(
                            running[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                        )
                    if tx % WARP_LANES == 0:
                        warp_prod[tx // WARP_LANES] = running[0]
                    T.sync_threads()
                    if tx == 0:
                        for w in T.serial(1, num_warps):
                            warp_prod[0] = warp_prod[0] * warp_prod[w]
                        out[row] = T.cast(warp_prod[0], dtype)

            return main
