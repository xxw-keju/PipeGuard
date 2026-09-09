    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def reduce_dh_recurrence_partials(
            dk_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
            dw_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, cid):
                for n, kk in T.Parallel(block_C, dim_k):
                    dk_acc = T.alloc_var(T.float32, init=0.0)
                    dw_acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        dk_acc += T.cast(
                            dk_partial[bid, hid, vid, cid * block_C + n, kk],
                            "float32",
                        )
                        dw_acc += T.cast(
                            dw_partial[bid, hid, vid, cid * block_C + n, kk],
                            "float32",
                        )
                    dk[bid, hid, cid * block_C + n, kk] = T.cast(dk_acc, dtype)
                    dw[bid, hid, cid * block_C + n, kk] = T.cast(dw_acc, dtype)

                for n in T.Parallel(block_C):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        acc += T.cast(dg_partial[bid, hid, vid, cid * block_C + n], "float32")
                    dg[bid, hid, cid * block_C + n] = T.cast(acc, dtype)

        return reduce_dh_recurrence_partials
