    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _dh_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        dht_shape = [batch, heads, dim_k, dim_v]
        dh_out_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dh0_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            dht: T.Tensor(dht_shape, accum_dtype),
            dh_out: T.Tensor(dh_out_shape, accum_dtype),
            dh0: T.Tensor(dh0_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_v_partitions, threads=threads) as bx:
                i_b = bx // (heads * num_v_partitions)
                i_h = (bx // num_v_partitions) % heads
                i_vp = bx % num_v_partitions
                v_offset = i_vp * dim_v_part

                dh_s = T.alloc_shared([dim_k, dim_v_part], accum_dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                do_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)

                # Load dht V-slice
                for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                    dh_s[i_k, i_v] = dht[i_b, i_h, i_k, v_offset + i_v]

                for t in T.Serial(num_chunks):
                    i_c = num_chunks - 1 - t
                    chunk_start = i_c * chunk_size

                    T.copy(
                        q[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                        q_s,
                        disable_tma=True,
                    )
                    T.copy(
                        do[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        do_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                        g_cumsum_s,
                        disable_tma=True,
                    )

                    g_last = T.alloc_fragment([dim_k], accum_dtype)
                    for i_k in T.Parallel(dim_k):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # q_gated (redundant across V-partitions)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E),
                            dtype,
                        )

                    # dh += scale * q_gated^T @ do_slice
                    dh_delta = T.alloc_fragment([dim_k, dim_v_part], accum_dtype)
                    T.fill(dh_delta, 0.0)
                    T.gemm(
                        q_gated_s, do_s, dh_delta, transpose_A=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = dh_s[i_k, i_v] + scale * dh_delta[i_k, i_v]

                    # Store dh BEFORE decay
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_out[i_b, i_c, i_h, i_k, v_offset + i_v] = dh_s[i_k, i_v]

                    # Decay for next (earlier) chunk
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = dh_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)

                # Write dh0
                if has_initial_state:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = dh_s[i_k, i_v]
                else:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = 0.0

        return _main
