    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):
        @T.prim_func
        def gla_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                h_tile_o = T.alloc_shared([k_tile, dim_v], dtype)
                v_shared = T.alloc_shared([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                # Preload q, k, gk into shared memory to avoid global access
                # in pipelined loop (fixes TileLang warp-specialization issue)
                q_shared = T.alloc_shared([dim_k], accum_dtype)
                k_shared = T.alloc_shared([dim_k], accum_dtype)
                gk_shared = T.alloc_shared([dim_k], accum_dtype)

                # Preload v into shared (for reuse in output and state update)
                for j in T.Parallel(dim_v):
                    v_shared[j] = T.cast(v[bid, hid, j], accum_dtype)

                # Preload q, k and gk into shared memory
                for i in T.Parallel(dim_k):
                    q_shared[i] = T.cast(q[bid, hid, i], accum_dtype)
                    k_shared[i] = T.cast(k[bid, hid, i], accum_dtype)
                    gk_shared[i] = T.cast(gk[bid, hid, i], accum_dtype)

                # Full-fp32 matvec.  TileLang 0.1.9 cannot reliably lower the
                # old tensor-core fragment copy here for fp16/bf16, and TF32
                # style matvec precision is too loose for recurrent decode.
                # TODO: restore a tensor-core fast path once fragment copies
                # lower reliably without sacrificing recurrent decode numerics.
                T.fill(sq_frag, 0.0)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile_o)
                    for kk in T.Serial(k_tile):
                        gk_val = gk_shared[kt * k_tile + kk]
                        alpha_i = T.exp2(gk_val * LOG2E)
                        q_gated = q_shared[kt * k_tile + kk] * alpha_i
                        for j in T.Parallel(dim_v):
                            sq_frag[j] = sq_frag[j] + q_gated * T.cast(h_tile_o[kk, j], accum_dtype)

                # Keep this scalar reduction separate from the matvec's
                # dim_v-parallel inner loop.
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j],
                        dtype,
                    )

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # NOTE: No intermediate variables inside T.Parallel to avoid BindNode issue
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E)
                            * T.cast(h_tile[kk, j], accum_dtype)
                            + k_shared[kt * k_tile + kk] * v_shared[j],
                            dtype,
                        )

        return gla_decode
