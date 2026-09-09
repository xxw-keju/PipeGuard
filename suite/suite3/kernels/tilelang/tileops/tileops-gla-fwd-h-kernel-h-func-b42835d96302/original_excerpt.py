    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _h_func(num_stages, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            initial_state: T.Tensor(init_state_shape, accum_dtype),
            h_out: T.Tensor(h_out_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = (bx // num_kv) % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)

                # Load initial state KV-slice
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_s[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]

                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        k[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            i_c * chunk_size : (i_c + 1) * chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    # Save pre-decay h KV-slice
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

                    # g_last from pre-computed cumsum
                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # Decay h
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)

                    # k_adj in fragment (RS GEMM: A=register, B=shared)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                            dtype,
                        )

                    # h += k_adj^T @ v_slice (RS GEMM)
                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]

                # Save final state KV-slice
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]

        return _main
