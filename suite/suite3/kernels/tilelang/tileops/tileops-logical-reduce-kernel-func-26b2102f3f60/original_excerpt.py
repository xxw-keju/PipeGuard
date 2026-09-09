    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Truthiness is decided at the load, so a tile costs one fragment.
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            val = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_val, "float32"),
                            )
                            bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                else:
                    # Load via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            bool_vals[i, j] = T.if_then_else(
                                T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                            )

                if op_kind == "any":
                    # any: result is 1 if max(bool_vals) == 1
                    T.reduce_max(bool_vals, result, dim=1)
                elif op_kind == "all":
                    # all: result is 1 if min(bool_vals) == 1
                    T.reduce_min(bool_vals, result, dim=1)
                else:
                    # count_nonzero: sum of bool values per row
                    T.reduce_sum(bool_vals, result, dim=1)

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i], out_dtype)
                else:
                    # Cast result to int8 (bool representation: 0 or 1)
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i] > 0.5, "int8")

                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            compute(x, out)

        return main
