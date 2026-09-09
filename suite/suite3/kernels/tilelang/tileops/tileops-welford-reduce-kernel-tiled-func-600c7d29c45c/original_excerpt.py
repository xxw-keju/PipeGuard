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
                        shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                        tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sum = T.alloc_fragment((block_m,), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sq = T.alloc_fragment((block_m,), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_v = T.alloc_fragment((block_m,), _out_dtype)
                        out_m = T.alloc_fragment((block_m,), _out_dtype)

                        T.fill(row_sum, 0.0)

                        # Pass 1: compute row sums for mean
                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                            T.reduce_sum(tile_f32, tile_sum, dim=1)
                            for i in T.Parallel(block_m):
                                row_sum[i] = row_sum[i] + tile_sum[i]

                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Pass 2: dedicated buffers to avoid TileLang aliasing
                        p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                        p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        T.fill(var_sum, 0.0)

                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                        p2_f32[i, j] - mean_val[i]
                                    )
                            T.reduce_sum(sq_diff, tile_sq, dim=1)
                            for i in T.Parallel(block_m):
                                var_sum[i] = var_sum[i] + tile_sq[i]

                        # Correct for padding: out-of-bound elements were filled
                        # with 0.0, so each contributes mean^2 to the sq_diff sum.
                        pad_count = total_cols - N
                        for i in T.Parallel(block_m):
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            if op_kind == "partials":
                                out_v[i] = corrected
                                out_m[i] = mean_val[i]
                            else:
                                out_v[i] = T.cast(corrected / float(N - correction), _out_dtype)
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
                        shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                        tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sum = T.alloc_fragment((block_m,), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sq = T.alloc_fragment((block_m,), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_local = T.alloc_fragment((block_m,), dtype)

                        T.fill(row_sum, 0.0)

                        # Pass 1: compute row sums for mean
                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                            T.reduce_sum(tile_f32, tile_sum, dim=1)
                            for i in T.Parallel(block_m):
                                row_sum[i] = row_sum[i] + tile_sum[i]

                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Pass 2: dedicated buffers
                        p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                        p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        T.fill(var_sum, 0.0)

                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                        p2_f32[i, j] - mean_val[i]
                                    )
                            T.reduce_sum(sq_diff, tile_sq, dim=1)
                            for i in T.Parallel(block_m):
                                var_sum[i] = var_sum[i] + tile_sq[i]

                        pad_count = total_cols - N
                        if op_kind == "var":
                            for i in T.Parallel(block_m):
                                corrected = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(corrected / float(N - correction), dtype)
                        else:  # std
                            for i in T.Parallel(block_m):
                                corrected = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(
                                    T.sqrt(corrected / float(N - correction)),
                                    dtype,
                                )

                        T.copy(out_local, out[pid_m * block_m])

            return main
