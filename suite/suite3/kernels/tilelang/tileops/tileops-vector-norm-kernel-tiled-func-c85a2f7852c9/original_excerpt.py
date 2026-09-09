    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                work = T.alloc_fragment((block_m, tile_n), work_dtype)
                acc = T.alloc_fragment((block_m,), work_dtype)
                tile_acc = T.alloc_fragment((block_m,), work_dtype)
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Zero is every kind's identity; for inf it is +0.0's bit pattern.
                T.fill(acc, 0)

                for t in T.Serial(num_tiles):
                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            work[i, j] = _prepared(
                                x,
                                pid_m * block_m + i,
                                t * tile_n + j,
                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                op_kind,
                            )

                    if op_kind == "inf":
                        T.reduce_max(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]

                if op_kind == "l2" and not partial:
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])

                for i in T.Parallel(block_m):
                    out_local[i] = _finished(acc[i], op_kind, out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main
