    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_H, block_N, num_stages, threads):
        shape_q = [batch, heads, dim]
        shape_kv = [seqlen_kv, groups, dim]
        kv_group_num = heads // groups

        valid_block_H = min(block_H, kv_group_num)

        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        @T.prim_func
        def gqa_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(batch, heads // valid_block_H, 1, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                seqlen_kv_b = real_seqlen_kv[bid]

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_kv_b, block_N)
                num_blockn_in_page = page_size // block_N

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = (
                        block_table[bid, page_idx] * num_blockn_in_page + block_idx_in_page
                    )

                    T.copy(
                        K[
                            blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        K_shared,
                    )
                    # Issue both K/V loads before QK so copies can overlap the GEMMs.
                    T.copy(
                        V[
                            blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        V_shared,
                    )
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (k * block_N + j < seqlen_kv_b), acc_s[i, j], -T.infinity(accum_dtype)
                        )
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_H):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

        return gqa_decode_no_split
