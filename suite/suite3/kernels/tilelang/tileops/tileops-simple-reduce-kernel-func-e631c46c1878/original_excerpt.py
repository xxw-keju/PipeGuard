        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), out_dtype)

                    if _needs_pad:
                        # Kernel-side boundary handling: element-wise load
                        # with T.if_then_else masking for padding columns
                        # and row-tail safety (M % block_m != 0).
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    T.cast(x[pid_m * block_m + i, j], "float32"),
                                    T.cast(_pad_val, "float32"),
                                )
                    else:
                        # Load via shared memory (fast vectorized path)
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        # Cast to fp32
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    if op_kind == "sum":
                        T.reduce_sum(x_f32, acc, dim=1)
                    elif op_kind == "mean":
                        T.reduce_sum(x_f32, acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] / float(N)
                    elif op_kind == "amax":
                        T.reduce_max(x_f32, acc, dim=1)
                    elif op_kind == "amin":
                        # Negate, reduce_max, negate back
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = -x_f32[i, j]
                        T.reduce_max(x_f32, acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = -acc[i]

                    # Cast back to output dtype
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], out_dtype)

                    T.copy(out_local, out[pid_m * block_m])

            return main
