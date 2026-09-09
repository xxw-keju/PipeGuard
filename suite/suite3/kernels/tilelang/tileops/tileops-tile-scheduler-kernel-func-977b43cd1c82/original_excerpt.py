    @tilelang.jit(out_idx=[1, 2, 3])
    def _func(threads: int):
        @T.prim_func
        def _sched_main(
            true_sizes: T.Tensor([num_experts], "int32"),
            tile_expert_ids: T.Tensor([max_tiles], "int32"),
            tile_row_offsets: T.Tensor([max_tiles], "int32"),
            total_tiles: T.Tensor([1], "int32"),
        ):
            with T.Kernel(T.ceildiv(max_tiles, threads), threads=threads) as (bx,):
                s_cum = T.alloc_shared([num_experts + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                _ex = T.alloc_local([1], "int32")
                _row = T.alloc_local([1], "int32")

                tx = T.get_thread_binding()

                if tx == 0:
                    _tiling.cumsum(true_sizes, s_cum)
                    if bx == 0:
                        total_tiles[0] = s_cum[num_experts]
                T.sync_threads()

                tile_id = bx * threads + tx
                if tile_id < max_tiles:
                    if tile_id < s_cum[num_experts]:
                        _tiling.decode(tile_id, s_cum, lo, hi, _ex, _row)
                        tile_expert_ids[tile_id] = _ex[0]
                        tile_row_offsets[tile_id] = _row[0]
                    else:
                        tile_expert_ids[tile_id] = T.int32(-1)
                        tile_row_offsets[tile_id] = T.int32(0)

        return _sched_main
