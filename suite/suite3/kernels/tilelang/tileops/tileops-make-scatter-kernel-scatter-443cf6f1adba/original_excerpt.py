    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(scatter_blocks, threads=_SCATTER_THREADS) as (bid,):
                tx = T.get_thread_binding()
                gid = bid * _SCATTER_THREADS + tx
                slot_buf = T.alloc_local([1], "int32")
                for i in T.serial(T.ceildiv(numel, total_scatter_threads)):
                    idx = gid + i * total_scatter_threads
                    if idx < numel:
                        eid = flat[idx]
                        # Store the slot before indexing with it: bounds-check
                        # lowering can otherwise duplicate the atomic.
                        slot_buf[0] = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                        slot = slot_buf[0]
                        sorted_token_ids[slot] = idx

        return _scatter_main
