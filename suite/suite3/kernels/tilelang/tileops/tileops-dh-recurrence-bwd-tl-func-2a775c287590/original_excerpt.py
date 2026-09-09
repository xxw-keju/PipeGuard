    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        correction_from_carry = _make_dh_correction_from_carry_macro(
            block_C,
            dim_k,
            BV,
            accum_dtype,
        )

        @T.prim_func
        def dh_recurrence_bwd_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            # Outputs
            dk_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
            dw_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                du_corr_c = T.alloc_shared([block_C, BV], dtype)
                # dh_buf carries gradient from the next chunk (backward)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_h_tmp = T.alloc_shared([dim_k, BV], dtype)
                d_g_pos = T.alloc_shared([block_C], dtype)
                d_g_last_partial = T.alloc_shared([dim_k], dtype)
                d_g_last_scalar1 = T.alloc_shared([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_shared([1], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                v_offset = vid * BV

                # Zero dh_buf (last chunk has no successor)
                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(
                        g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True
                    )
                    T.copy(
                        k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        k_c,
                        disable_tma=True,
                    )
                    T.copy(
                        v_new[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(S[bid, hid, t_bwd, :, v_offset : v_offset + BV], h_c, disable_tma=True)
                    T.copy(
                        dh_local[bid, hid, t_bwd, :, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )

                    # dh = dh_local + dh_buf * exp(g_last)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(
                            g_c[block_C - 1] * LOG2E
                        )

                    correction_from_carry(
                        g_c,
                        k_c,
                        v_new_c,
                        h_c,
                        dh_buf,
                        k_scaled,
                        dP,
                        dg_c,
                        du_corr_c,
                        dh_h_tmp,
                        du_corr_frag,
                        dP_frag,
                        d_g_pos,
                        d_g_last_partial,
                        d_g_last_scalar1,
                        d_g_last_scalar2,
                        dk_corr_partial,
                        du_corr,
                        dg_corr_partial,
                        dw_corr_partial,
                        bid,
                        hid,
                        vid,
                        t_bwd * block_C,
                        v_offset,
                    )

                    # Carry dh to next iteration
                    T.copy(dh_frag, dh_buf)

        return dh_recurrence_bwd_kernel
