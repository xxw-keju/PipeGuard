    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3"],
    )
    def _decode_func(num_stages, threads=128):
        @T.prim_func
        def gla_decode_fp32(
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
                # Fragment accumulator for S @ q_gated (full fp32, no TF32)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                # Preload q, k, v, gk into shared memory to avoid global access
                # in pipelined loop (fixes TileLang warp-specialization issue)
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                gk_shared = T.alloc_shared([dim_k], dtype)

                # Preload tensors into shared memory
                for i in T.Parallel(dim_k):
                    q_shared[i] = q[bid, hid, i]
                    k_shared[i] = k[bid, hid, i]
                    gk_shared[i] = gk[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_shared[i] = v[bid, hid, i]

                T.fill(sq_frag, 0.0)

                # === Pass 1: Element-wise matvec (full fp32 precision) ===
                # S @ q_gated where q_gated = q * exp(gk)
                for kk in T.Serial(dim_k):
                    gk_val = gk_shared[kk]
                    alpha_i = T.exp2(gk_val * LOG2E)
                    q_gated = q_shared[kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]

                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j]

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # NOTE: No intermediate variables inside T.Parallel to avoid BindNode issue
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * h_tile[kk, j]
                            + k_shared[kt * k_tile + kk] * v_shared[j]
                        )

        return gla_decode_fp32
