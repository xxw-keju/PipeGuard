    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        per_thread = max(1, block_m // threads)
        span = per_thread * threads
        # Staging needs one contiguous run, no masked tail, and a tile within budget.
        stage = (
            span == block_m
            and M % span == 0
            and inner_stride % span == 0
            and N * span * elem_bytes <= SHARED_MEMORY_BUDGET_BYTES
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, span), threads=threads) as pid:
                tx = T.get_thread_binding()
                tile = T.alloc_shared((N, span if stage else 1), dtype)
                keys = T.alloc_local((per_thread,), "int32")
                indices = T.alloc_local((per_thread,), "int32")
                candidate = T.alloc_local((1,), "int32")

                if stage:
                    outer = (pid * span) // inner_stride
                    inner = (pid * span) % inner_stride
                    for index, column in T.Parallel(N, span):
                        tile[index, column] = x[
                            outer * N * inner_stride + index * inner_stride + inner + column
                        ]
                    T.sync_threads()

                for slot in T.serial(per_thread):
                    ops.set_identity(keys, indices, slot)

                for index in T.serial(N):
                    for slot in T.serial(per_thread):
                        column = slot * threads + tx
                        if stage:
                            candidate[0] = key_of(tile[index, column])
                            ops.advance(keys, indices, slot, candidate[0], index)
                        else:
                            row = pid * span + column
                            if row < M:
                                offset = (
                                    (row // inner_stride) * N * inner_stride
                                    + index * inner_stride
                                    + row % inner_stride
                                )
                                candidate[0] = key_of(x[offset])
                                ops.advance(keys, indices, slot, candidate[0], index)

                for slot in T.serial(per_thread):
                    row = pid * span + slot * threads + tx
                    if row < M:
                        out[row] = T.cast(indices[slot], "int64")

        return main
