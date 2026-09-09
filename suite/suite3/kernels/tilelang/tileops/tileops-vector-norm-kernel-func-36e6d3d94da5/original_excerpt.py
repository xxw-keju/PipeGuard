    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                work = T.alloc_fragment((block_m, N_padded), work_dtype)
                acc = T.alloc_fragment((block_m,), work_dtype)
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Loaded straight into the working fragment, with no staging buffer.
                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        if _needs_pad:
                            in_bounds = T.And(pid_m * block_m + i < M, j < N)
                        else:
                            in_bounds = pid_m * block_m + i < M
                        work[i, j] = _prepared(x, pid_m * block_m + i, j, in_bounds, op_kind)

                if op_kind == "l1":
                    T.reduce_sum(work, acc, dim=1)
                elif op_kind == "l2":
                    T.reduce_sum(work, acc, dim=1)
                    if not partial:
                        for i in T.Parallel(block_m):
                            acc[i] = T.sqrt(acc[i])
                else:
                    T.reduce_max(work, acc, dim=1)

                for i in T.Parallel(block_m):
                    out_local[i] = _finished(acc[i], op_kind, out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main
