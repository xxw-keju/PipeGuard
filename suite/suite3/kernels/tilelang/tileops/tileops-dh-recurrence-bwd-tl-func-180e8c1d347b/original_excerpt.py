    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_recurrence_bwd_kernel(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            # Outputs
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                dh_loc = T.alloc_shared([dim_k, dim_v], accum_dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dh_buf = T.alloc_shared([dim_k, dim_v], dtype)
                dh_fp32 = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                wk_dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                k_dh_shared = T.alloc_shared([block_C, dim_v], dtype)

                for i, j in T.Parallel(dim_k, dim_v):
                    dh_fp32[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(
                        k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        k_c,
                        disable_tma=True,
                    )
                    T.copy(
                        w[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        w_c,
                        disable_tma=True,
                    )
                    T.copy(
                        v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(dh_local[bid, hid, t_bwd, :, :], dh_loc, disable_tma=True)

                    T.copy(dh_fp32, dh_buf)

                    # du_corr = k @ dh_buf; also save for wk_dh
                    T.clear(du_corr_frag)
                    T.gemm(k_c, dh_buf, du_corr_frag)
                    T.copy(
                        du_corr_frag,
                        du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        disable_tma=True,
                    )
                    T.copy(du_corr_frag, k_dh_shared)

                    # dk_corr = v_new @ dh_buf^T
                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr[bid, hid, t_bwd * block_C + n, kk] = dP[n, kk]

                    # dh = dh_local + (I - W^T @ K) @ dh_future
                    #    = dh_local + dh_future - W^T @ (K @ dh_future)
                    T.clear(wk_dh_frag)
                    T.gemm(w_c, k_dh_shared, wk_dh_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_fp32[i, j] = dh_fp32[i, j] + dh_loc[i, j] - wk_dh_frag[i, j]

        return dh_recurrence_bwd_kernel
