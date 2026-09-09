    @tilelang.jit(out_idx=[2])
    def _func():
        ops = _make_pair_ops(op_kind, N)

        @T.prim_func
        def main(
            partial_keys: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, rows_per_block), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * rows_per_block + tx // WARP_LANES
                lane = tx % WARP_LANES
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.set_identity(best_key, best_index, 0)

                for step in T.serial(partials_per_lane):
                    partial = step * WARP_LANES + lane
                    if row < M and partial < ctas_per_row:
                        ops.update(
                            best_key,
                            best_index,
                            0,
                            partial_keys[row * ctas_per_row + partial],
                            partial_indices[row * ctas_per_row + partial],
                        )

                ops.warp_reduce(best_key, best_index, WARP_LANES.bit_length() - 1, WARP_LANES)
                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main
