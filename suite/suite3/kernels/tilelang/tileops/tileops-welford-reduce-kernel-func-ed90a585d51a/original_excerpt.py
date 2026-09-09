        @tilelang.jit(out_idx=out_idx)
        def _func(block_m, threads):
            if _pair:

                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out_var: T.Tensor[(M,), _out_dtype],
                    out_mean: T.Tensor[(M,), _out_dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                        x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_v = T.alloc_fragment((block_m,), _out_dtype)
                        out_m = T.alloc_fragment((block_m,), _out_dtype)

                        if _needs_pad:
                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.if_then_else(
                                        T.And(pid_m * block_m + i < M, j < N),
                                        T.cast(x[pid_m * block_m + i, j], "float32"),
                                        T.cast(0.0, "float32"),
                                    )
                        else:
                            T.copy(x[pid_m * block_m, 0], shared_buf)

                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(x_f32, row_sum, dim=1)
                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Variance: sum((x - mean)^2) / (N - correction)
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                dev = x_f32[i, j] - mean_val[i]
                                sq_diff[i, j] = dev * dev

                        T.reduce_sum(sq_diff, var_sum, dim=1)

                        # Correct for padding: padded elements contribute mean^2 each
                        pad_count = N_padded - N
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            if op_kind == "partials":
                                out_v[i] = corrected_sum
                                out_m[i] = mean_val[i]
                            else:
                                out_v[i] = T.cast(corrected_sum / float(N - correction), _out_dtype)
                                out_m[i] = T.cast(mean_val[i], _out_dtype)

                        T.copy(out_v, out_var[pid_m * block_m])
                        T.copy(out_m, out_mean[pid_m * block_m])

            else:
                # std or var (single output)
                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out: T.Tensor[(M,), dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                        x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_local = T.alloc_fragment((block_m,), dtype)

                        if _needs_pad:
                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.if_then_else(
                                        T.And(pid_m * block_m + i < M, j < N),
                                        T.cast(x[pid_m * block_m + i, j], "float32"),
                                        T.cast(0.0, "float32"),
                                    )
                        else:
                            T.copy(x[pid_m * block_m, 0], shared_buf)

                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(x_f32, row_sum, dim=1)
                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Variance
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                dev = x_f32[i, j] - mean_val[i]
                                sq_diff[i, j] = dev * dev

                        T.reduce_sum(sq_diff, var_sum, dim=1)

                        pad_count = N_padded - N
                        if op_kind == "var":
                            for i in T.Parallel(block_m):
                                corrected_sum = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(corrected_sum / float(N - correction), dtype)
                        else:  # std
                            for i in T.Parallel(block_m):
                                corrected_sum = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(
                                    T.sqrt(corrected_sum / float(N - correction)), dtype
                                )

                        T.copy(out_local, out[pid_m * block_m])

            return main
