    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * block_m + tx // lanes
                lane = tx % lanes

                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = iteration * items_per_iteration + accumulator * lanes + lane
                        if row < M and index < N:
                            candidate[0] = key_of(x[row, index])
                            ops.advance(keys, indices, accumulator, candidate[0], index)

                ops.merge_accumulators(keys, indices, best_key, best_index)
                ops.warp_reduce(best_key, best_index, log_lanes, lanes)

                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main
