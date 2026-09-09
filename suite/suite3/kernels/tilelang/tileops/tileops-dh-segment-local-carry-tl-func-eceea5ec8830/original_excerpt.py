    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_segment_local_carry_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            segment_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_segments, dim_k, BV], dtype
            ),
            dh_carry_after: T.Tensor([batch, head, num_v_tiles, num_chunks, dim_k, BV], dtype),
        ):
            with T.Kernel(num_v_tiles, num_segments, batch * head, threads=threads) as (
                vid,
                sid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid - bid * head
                g_c = T.alloc_shared([block_C], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                carry = T.alloc_shared([dim_k, BV], dtype)
                carry_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV

                T.copy(
                    segment_carry_after[bid, hid, vid, sid, :, :],
                    carry,
                    disable_tma=True,
                )

                for step in T.Serial(segment_chunks):
                    local_idx = segment_chunks - 1 - step
                    cid = sid * segment_chunks + local_idx
                    T.copy(
                        carry,
                        dh_carry_after[bid, hid, vid, cid, :, :],
                        disable_tma=True,
                    )
                    T.copy(
                        g[bid, hid, cid * block_C : (cid + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, cid, :, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    alpha = T.exp2(g_c[block_C - 1] * LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        carry_frag[i, j] = dh_loc[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)

        return dh_segment_local_carry_kernel
