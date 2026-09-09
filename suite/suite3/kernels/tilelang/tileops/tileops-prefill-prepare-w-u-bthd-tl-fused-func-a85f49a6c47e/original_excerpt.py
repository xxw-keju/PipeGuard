    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.prim_func
        def prefill_prepare_w_u_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (
                bid,
                hid,
                by,
            ):
                base = by * block_C
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                g_pos_shared = T.alloc_shared([block_C], accum_dtype)
                g_neg_shared = T.alloc_shared([block_C], accum_dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                for i, d in T.Parallel(block_C, dim_k):
                    k_shared[i, d] = k[bid, base + i, hid, d]
                for i in T.Parallel(block_C):
                    g_shared[i] = g[bid, base + i, hid]
                    beta_shared[i] = beta[bid, base + i, hid]

                for i, d in T.Parallel(block_C, dim_k):
                    k_beta_shared[i, d] = k_shared[i, d] * beta_shared[i]
                for i in T.Parallel(block_C):
                    g_pos_shared[i] = T.exp2(g_shared[i] * LOG2E)
                    g_neg_shared[i] = T.exp2(-g_shared[i] * LOG2E)
                T.clear(gram_frag)
                T.gemm(k_beta_shared, k_shared, gram_frag, transpose_B=True)

                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j] * g_pos_shared[i] * g_neg_shared[j],
                        T.float32(0.0),
                    )
                T.clear(S_shared)
                for i in T.Parallel(block_C):
                    S_shared[i, i] = T.float32(1.0)

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                for i, d in T.Parallel(block_C, dim_k):
                    w[bid, base + i, hid, d] = w_frag[i, d]

                for i, d in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, d] = v[bid, base + i, hid, d] * beta_shared[i]
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                for i, d in T.Parallel(block_C, dim_v):
                    u[bid, base + i, hid, d] = u_frag[i, d]

        return prefill_prepare_w_u_bthd
