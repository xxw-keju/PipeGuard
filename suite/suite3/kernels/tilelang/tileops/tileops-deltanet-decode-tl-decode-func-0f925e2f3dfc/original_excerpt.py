    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):
        @T.macro
        def _decode_body(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                h_tile_o = T.alloc_shared([k_tile, dim_v], dtype)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                beta_val = T.cast(beta[bid, hid], accum_dtype)

                # Full-fp32 matvecs.  TileLang 0.1.9 cannot reliably lower
                # the old tensor-core fragment copy here for fp16/bf16.
                # TODO: restore a tensor-core fast path once fragment copies
                # lower reliably without sacrificing recurrent decode numerics.
                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile_o)
                    for kk in T.Serial(k_tile):
                        k_val = T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype)
                        q_val = T.cast(q[bid, hid, kt * k_tile + kk], accum_dtype)
                        for j in T.Parallel(dim_v):
                            h_val = T.cast(h_tile_o[kk, j], accum_dtype)
                            sk_frag[j] = sk_frag[j] + k_val * h_val
                            sq_frag[j] = sq_frag[j] + q_val * h_val

                # q . k is a scalar reduction, so keep it separate from the
                # matvec loop whose inner work is parallelized over dim_v.
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += T.cast(q[bid, hid, kk], accum_dtype) * T.cast(
                        k[bid, hid, kk], accum_dtype
                    )

                # v_new = beta * (v - S @ k) (no alpha)
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * (T.cast(v[bid, hid, j], accum_dtype) - sk_frag[j])

                # o = S @ q + (q . k) * v_new (no alpha scaling)
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(sq_frag[j] + qk_dot[0] * v_new[j], dtype)

                # === Pass 2: State update with async prefetch ===
                # new_state = S + outer(k, v_new) (no alpha decay)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            T.cast(h_tile[kk, j], accum_dtype)
                            + T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype) * v_new[j],
                            dtype,
                        )

        @T.prim_func
        def deltanet_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            _decode_body(q, k, v, beta, state, o, new_state)

        return deltanet_decode
