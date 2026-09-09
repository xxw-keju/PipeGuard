    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_H, block_N, num_split, num_stages, threads):
        shape_q = [batch, heads, dim]
        shape_kv = [seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        kv_group_num = heads // groups

        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)

        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_H, block_N)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_H, dim)

        @T.macro
        def _gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            split_length: T.Tensor([batch, num_split], "int32"),
        ):
            with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (
                bx,
                by,
                bz,
            ):
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

                split_length_shared = T.alloc_shared([num_split], "int32")
                bid = bx
                hid = by
                sid = bz
                T.copy(split_length[bid, :], split_length_shared, disable_tma=True)
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                seqlen_kv_b = real_seqlen_kv[bid]

                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Per-batch loop_range: only iterate blocks that are within real_seqlen_kv[bid],
                # so shorter batches (e.g. batch1 with 2048) don't run empty splits -> all -inf -> NaN
                start_block_sid = T.if_then_else(
                    sid > 0, split_length_shared[sid - 1] // block_N, 0
                )
                end_block_valid = T.ceildiv(seqlen_kv_b, block_N)
                blocks_valid_this_split = end_block_valid - start_block_sid
                blocks_in_split = T.if_then_else(
                    sid > 0,
                    T.ceildiv(split_length_shared[sid] - split_length_shared[sid - 1], block_N),
                    T.ceildiv(split_length_shared[0], block_N),
                )
                loop_range = T.if_then_else(
                    blocks_valid_this_split <= 0,
                    0,
                    T.if_then_else(
                        blocks_valid_this_split <= blocks_in_split,
                        blocks_valid_this_split,
                        blocks_in_split,
                    ),
                )

                num_blockn_in_page = page_size // block_N
                offset = 0 if sid == 0 else split_length_shared[sid - 1] // block_N
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    k_global = k
                    k_global += offset
                    page_idx = k_global // num_blockn_in_page
                    block_idx_in_page = k_global % num_blockn_in_page
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
                    start_sid = T.if_then_else(sid > 0, split_length_shared[sid - 1], 0)
                    for i, j in T.Parallel(block_H, block_N):
                        logical_pos = start_sid + k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            logical_pos < seqlen_kv_b, acc_s[i, j], -T.infinity(accum_dtype)
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
                    # When loop_range was 0 (split entirely beyond real_seqlen_kv), logsum=0 -> avoid 0/0
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_H):
                    # Avoid log2(0)=-inf when logsum=0; glse=-inf is ok in combine (weight 0)
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(
                    O_shared,
                    Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :],
                )

        @T.macro
        def combine(
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                #
                glse_vec = T.alloc_fragment([num_split], dtype)
                for k in T.Parallel(num_split):
                    glse_vec[k] = glse[bz, by, k]
                lse_max = T.alloc_fragment([1], accum_dtype)
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(glse_vec, lse_max, dim=0, clear=False)

                #
                lse_logsum = T.alloc_local([1], accum_dtype)
                lse_logsum[0] = 0
                for k in T.serial(num_split):
                    lse_logsum[0] += T.exp2(glse[bz, by, k] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]

                #
                o_accum = T.alloc_fragment([dim], accum_dtype)
                T.clear(o_accum)
                for k in T.serial(num_split):
                    w = T.exp2(glse[bz, by, k] - lse_logsum[0])
                    for i in T.Parallel(dim):
                        o_accum[i] += Output_partial[bz, by, k, i] * w
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum[i]

        @T.prim_func
        def gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            split_length: T.Tensor([batch, num_split], "int32"),
            Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_split(
                Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, split_length
            )
            combine(glse, Output_partial, Output)

        return gqa_decode_split
