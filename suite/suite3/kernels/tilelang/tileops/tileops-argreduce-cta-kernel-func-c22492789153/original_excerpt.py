    @tilelang.jit(out_idx=[1])
    def _func(threads: int):
        num_warps = threads // WARP_LANES
        iterations = (N + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        held = threads * FRAGMENT_ELEMS_PER_THREAD >= N
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        block_reduce = _make_block_reduce(ops, num_warps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                warp_keys = T.alloc_shared((num_warps,), "int32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                row_frag = T.alloc_fragment((1, N if held else 1), dtype)
                # Four slots either way, so both paths share the block reduction below.
                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                if held:
                    T.copy(x[row : row + 1, :], row_frag)
                    # The fragment order is unstated, so this merge compares indices.
                    for _, column in T.Parallel(1, N):
                        candidate[0] = key_of(row_frag[0, column])
                        ops.update(keys, indices, 0, candidate[0], column)
                else:
                    for iteration in T.serial(iterations):
                        for accumulator in T.serial(_NUM_ACCUMULATORS):
                            index = (
                                iteration * threads * _NUM_ACCUMULATORS + accumulator * threads + tx
                            )
                            if index < N:
                                candidate[0] = key_of(x[row, index])
                                ops.advance(keys, indices, accumulator, candidate[0], index)

                block_reduce(
                    keys,
                    indices,
                    best_key,
                    best_index,
                    warp_keys,
                    warp_indices,
                    tx,
                )
                if tx == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main
