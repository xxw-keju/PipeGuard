    @tilelang.jit(out_idx=[1, 2])
    def _func(threads: int, ctas_per_row: int):
        chunk_size = (N + ctas_per_row - 1) // ctas_per_row
        num_partials = M * ctas_per_row
        num_warps = threads // WARP_LANES
        iterations = (chunk_size + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        block_reduce = _make_block_reduce(ops, num_warps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            partial_keys: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
        ):
            with T.Kernel(num_partials, threads=threads) as partial:
                tx = T.get_thread_binding()
                row = partial // ctas_per_row
                split = partial % ctas_per_row
                chunk_start = split * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, N)

                warp_keys = T.alloc_shared((num_warps,), "int32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = (
                            chunk_start
                            + iteration * threads * _NUM_ACCUMULATORS
                            + accumulator * threads
                            + tx
                        )
                        if index < chunk_end:
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
                    partial_keys[partial] = best_key[0]
                    partial_indices[partial] = best_index[0]

        return main
