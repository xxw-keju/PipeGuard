    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_sw_fwd_varlen_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        q_shape = (total_q, heads, dim)
        kv_shape = (total_k, heads_kv, dim)
        apply_mask = _make_apply_mask(
            is_causal,
            has_window,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
        )
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.macro
        def mma0(
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            q_shared: T.SharedBuffer([block_m, dim], dtype),  # type: ignore
            k_shared: T.SharedBuffer([block_n, dim], dtype),  # type: ignore
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),  # type: ignore
            k_idx: T.int32,  # type: ignore
            bx: T.int32,  # type: ignore
            by: T.int32,  # type: ignore
            kv_start: T.int32,  # type: ignore
            q_len: T.int32,  # type: ignore
            kv_len: T.int32,  # type: ignore
            offset: T.int32,  # type: ignore
        ) -> None:
            T.copy(
                k[kv_start + k_idx * block_n : kv_start + (k_idx + 1) * block_n, by // groups, :],
                k_shared,
            )
            apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            v_shared: T.SharedBuffer([block_n, dim], dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),  # type: ignore
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),  # type: ignore
            k_idx: T.int32,  # type: ignore
            by: T.int32,  # type: ignore
            kv_start: T.int32,  # type: ignore
        ) -> None:
            T.copy(
                v[kv_start + k_idx * block_n : kv_start + (k_idx + 1) * block_n, by // groups, :],
                v_shared,
            )
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_varlen_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})

                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_k[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_k[bz + 1] - kv_start
                offset = kv_len - q_len

                T.copy(q[q_start + bx * block_m : q_start + (bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(kv_len, offset + (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(kv_len, offset + (bx + 1) * block_m + window_size_right), block_n
                    )
                else:
                    k_end = T.ceildiv(kv_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(0, offset + bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(
                        k, q_shared, k_shared, acc_s, k_idx, bx, by, kv_start, q_len, kv_len, offset
                    )
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, kv_start)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
                T.sync_threads(3, threads)
                T.copy(acc_o, o_shared)
                T.sync_threads(3, threads)
                for i, j in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by, j] = o_shared[i, j]
                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        lse[by, q_start + bx * block_m + i] = logsum[i]

        return _gqa_sw_fwd_varlen_wgmma_pipelined_main
