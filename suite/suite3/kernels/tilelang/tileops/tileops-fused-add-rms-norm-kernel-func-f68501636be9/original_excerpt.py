    @tilelang.jit(out_idx=[3, 4])
    def _func(block_m, threads):
        # Forming the sum in shared rather than a second row-wide fragment keeps
        # bfloat16 off the register limit, but only pays when the registers it
        # frees buy resident blocks: a single-row call measured 0.94x.
        sum_in_shared = -(-M // block_m) >= sm_count
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if sum_in_shared:
                    shared_x = T.alloc_shared((block_m, N_padded), dtype)
                    shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                xsq_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                if sum_in_shared:
                    T.copy(x[pid_m * block_m, 0], shared_x)
                    T.copy(residual[pid_m * block_m, 0], shared_r)
                    for i, j in T.Parallel(block_m, N_padded):
                        shared_x[i, j] = T.cast(
                            T.cast(shared_x[i, j], "float32") + T.cast(shared_r[i, j], "float32"),
                            dtype,
                        )
                    T.sync_threads()
                    T.copy(shared_x, residual_out[pid_m * block_m, 0])
                    T.copy(shared_x, x_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    for i, j in T.Parallel(block_m, N_padded):
                        if (not row_guard) or pid_m * block_m + i < M:
                            y[pid_m * block_m + i, j] = T.cast(
                                T.cast(x_local[i, j], "float32")
                                * rrms[i]
                                * T.cast(weight[j], "float32"),
                                dtype,
                            )
                else:
                    r_local = T.alloc_fragment((block_m, N_padded), dtype)

                    if row_guard:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                        for i, j in T.Parallel(block_m, N_padded):
                            r_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                residual[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = x[pid_m * block_m + i, j]
                        for i, j in T.Parallel(block_m, N_padded):
                            r_local[i, j] = residual[pid_m * block_m + i, j]

                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.cast(x_local[i, j], "float32") + T.cast(
                            r_local[i, j], "float32"
                        )

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    for i, j in T.Parallel(block_m, N_padded):
                        if (not row_guard) or pid_m * block_m + i < M:
                            y[pid_m * block_m + i, j] = T.cast(
                                T.cast(x_local[i, j], "float32")
                                * rrms[i]
                                * T.cast(weight[j], "float32"),
                                dtype,
                            )

                    for i, j in T.Parallel(block_m, N_padded):
                        if (not row_guard) or pid_m * block_m + i < M:
                            residual_out[pid_m * block_m + i, j] = x_local[i, j]

        return main
