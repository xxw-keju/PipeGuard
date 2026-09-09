    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        # A non-aligned D would read and write columns >= D unless masked.
        masked = D_padded != D
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

                # No-affine output: y = (x - mean) * rstd.
                if guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        if T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast(
                                (
                                    T.cast(
                                        centered_row[i, j] if register_direct else shared_buf[i, j],
                                        "float32",
                                    )
                                    - mean_val[i]
                                )
                                * rstd[i],
                                dtype,
                            )
                else:
                    # Not guarded means D_padded == D, which is a register-held row.
                    for i, j in T.Parallel(block_m, D_padded):
                        y[pid_m * block_m + i, j] = T.cast(
                            (centered_row[i, j] - mean_val[i]) * rstd[i], dtype
                        )

        return main
