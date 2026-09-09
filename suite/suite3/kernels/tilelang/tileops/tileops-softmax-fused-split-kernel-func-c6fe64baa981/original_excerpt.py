    @tilelang.jit(out_idx=[3])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                held = T.alloc_fragment((1, seg_n), "float32")
                shifted = T.alloc_fragment((1, seg_n), "float32")
                seg_m = T.alloc_fragment((1,), "float32")
                seg_s = T.alloc_fragment((1,), "float32")
                part_max = T.alloc_fragment((1, threads), "float32")
                part_sum = T.alloc_fragment((1, threads), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")
                row_scale = T.alloc_local((1,), "float32")
                shift = T.alloc_local((1,), "float32")

                for _, j in T.Parallel(1, seg_n):
                    held[0, j] = T.if_then_else(
                        pid_s * seg_n + j < N,
                        T.cast(x[pid_m, pid_s * seg_n + j], "float32"),
                        -T.infinity("float32"),
                    )
                T.fill(seg_m, -T.infinity("float32"))
                T.reduce_max(held, seg_m, dim=1, clear=False)
                # A segment of only -inf shifts by zero instead of by its own
                # -inf, which would leave every lane the NaN of exp(-inf - -inf).
                # Shifting by zero leaves exp(-inf) = 0, so the segment sums to
                # zero and contributes nothing; the test is the row's, so it is
                # taken once here rather than at every element.
                shift[0] = T.if_then_else(
                    seg_m[0] == -T.infinity("float32"), T.cast(0.0, "float32"), seg_m[0]
                )
                for _, j in T.Parallel(1, seg_n):
                    shifted[0, j] = T.exp(held[0, j] - shift[0])
                T.reduce_sum(shifted, seg_s, dim=1)
                seg_max[pid_m * num_segs + pid_s] = seg_m[0]
                seg_sum[pid_m * num_segs + pid_s] = seg_s[0]

                T.sync_grid()

                fold(seg_max, seg_sum, pid_m * num_segs, part_max, part_sum, row_max, row_sum)

                if op_kind == "softmax":
                    row_scale[0] = T.exp(seg_m[0] - row_max[0]) / row_sum[0]
                else:
                    row_scale[0] = row_max[0] + T.log(row_sum[0])

                # Indexed with the expression rather than a name bound to it:
                # TileLang's parallel-loop verifier does not substitute the
                # binding, and reads the store as j-independent.
                for _, j in T.Parallel(1, seg_n):
                    with T.If(pid_s * seg_n + j < N):  # noqa: SIM117
                        with T.Then():
                            if op_kind == "softmax":
                                y[pid_m, pid_s * seg_n + j] = T.cast(
                                    shifted[0, j] * row_scale[0], dtype
                                )
                            else:
                                y[pid_m, pid_s * seg_n + j] = T.cast(
                                    held[0, j] - row_scale[0], dtype
                                )

        return main
