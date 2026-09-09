    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):
        # A non-aligned D would read and write columns >= D unless masked.
        masked = D_padded != D
        # One channel owns the whole row exactly when a group holds one channel.
        row_constant_affine = channels_per_group == 1
        register_direct = _holds_row_in_registers(D, D_padded)
        # A tail row block runs past the end unless every index is guarded.
        guarded = masked or M % block_m != 0
        if register_direct:
            row_reduce = make_shifted_row_reduce(block_m, D, eps)
        else:
            row_reduce = make_row_reduce(block_m, D, D_padded, eps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if register_direct:
                    centered_row = T.alloc_fragment((block_m, D_padded), "float32")
                    squares = T.alloc_fragment((block_m, D_padded), "float32")
                    shift = T.alloc_fragment((block_m,), "float32")
                    acc_squares = T.alloc_fragment((block_m,), "float32")
                else:
                    shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")
                if row_constant_affine:
                    scale = T.alloc_fragment((block_m,), "float32")
                    bias_row = T.alloc_fragment((block_m,), "float32")

                # Ahead of the row load, so the two reads are in flight while
                # it runs. ``scale`` holds the weight until rstd exists.
                if row_constant_affine:
                    for i in T.Parallel(block_m):
                        c = (pid_m * block_m + i) % num_groups
                        scale[i] = T.cast(weight[c], "float32")
                        bias_row[i] = T.cast(bias[c], "float32")

                if register_direct:
                    # The shift the one-pass reduction needs. A tail block's
                    # row index is clamped to stay inside the tensor.
                    for i in T.Parallel(block_m):
                        shift[i] = T.cast(x[T.min(pid_m * block_m + i, M - 1), 0], "float32")
                    if guarded:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < D),
                                T.cast(x[pid_m * block_m + i, j], "float32") - shift[i],
                                T.cast(0.0, "float32"),
                            )
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    else:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.cast(x[pid_m * block_m + i, j], "float32") - shift[i]
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    row_reduce(centered_row, squares, acc, acc_squares, mean_val, rstd)
                else:
                    # A padded wide row keeps its input in shared memory for the
                    # output pass: the reduction overwrites the fp32 copy with
                    # the centered squares.
                    for i, j in T.Parallel(block_m, D_padded):
                        shared_buf[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < D),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                    row_reduce(x_f32, acc, mean_val, rstd)

                # --- Output: y = (x - mean) * rstd * weight[c] + bias[c] ---
                if row_constant_affine:
                    # A group of one channel gives the whole row one weight and
                    # one bias, so the two gathers leave the element loop.
                    for i in T.Parallel(block_m):
                        scale[i] = scale[i] * rstd[i]
                    for i, j in T.Parallel(block_m, D_padded):
                        if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast(
                                (
                                    T.cast(
                                        centered_row[i, j] if register_direct else shared_buf[i, j],
                                        "float32",
                                    )
                                    - mean_val[i]
                                )
                                * scale[i]
                                + bias_row[i],
                                dtype,
                            )
                elif guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        if T.And(pid_m * block_m + i < M, j < D):
                            c = _channel_of(
                                pid_m * block_m + i,
                                j,
                                num_groups,
                                channels_per_group,
                                spatial_size,
                            )
                            y[pid_m * block_m + i, j] = (
                                T.cast(
                                    centered_row[i, j] if register_direct else shared_buf[i, j],
                                    "float32",
                                )
                                - mean_val[i]
                            ) * rstd[i] * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")
                else:
                    # Not guarded means D_padded == D, which is a register-held row.
                    for i, j in T.Parallel(block_m, D_padded):
                        c = _channel_of(
                            pid_m * block_m + i,
                            j,
                            num_groups,
                            channels_per_group,
                            spatial_size,
                        )
                        y[pid_m * block_m + i, j] = (centered_row[i, j] - mean_val[i]) * rstd[
                            i
                        ] * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")

        return main
