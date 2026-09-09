    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scan(threads: int):
        scan = _make_tight_scan_body(numel, num_experts, top_k, threads)

        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),
            physical_ends: T.Tensor([num_experts], "int32"),
            permuted_idx: T.Tensor([numel], "int32"),
            inverse_indices: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
        ):
            with T.Kernel(1, threads=threads) as (_,):
                counts = T.alloc_shared([num_experts], "int32")
                offsets = T.alloc_shared([num_experts + 1], "int64")
                slot_buf = T.alloc_local([1], "int32")
                scan(
                    flat_ids,
                    physical_ends,
                    permuted_idx,
                    inverse_indices,
                    write_offsets,
                    counts,
                    offsets,
                    slot_buf,
                )

        return _scan_main
