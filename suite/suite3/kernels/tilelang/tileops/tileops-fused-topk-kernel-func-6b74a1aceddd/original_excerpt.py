    @tilelang.jit(out_idx=[])
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = WARP_LANES
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)  # ceildiv(E, 32)
        LOG_WARP = int(math.log2(WARP_SIZE))  # = 5 for WARP_SIZE=32
        HALF_WARP = WARP_SIZE // 2  # = 16
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)  # ceildiv(T, TPB)

        if with_correction_bias:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], "float32"),
                correction_bias: T.Tensor([num_experts], "float32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id

                    # my_biased only decides selection; my_scores is what gets written.
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    my_biased = T.alloc_local([ELEMS_PER_THREAD], "float32")

                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                val = my_scores[j]
                                sig_val = T.float32(1) / (T.float32(1) + T.exp(-val))
                                my_scores[j] = sig_val
                                my_biased[j] = sig_val + correction_bias[expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")
                                my_biased[j] = -T.infinity("float32")

                        l_best_val = T.alloc_var(T.float32)  # biased (for selection)
                        l_best_orig = T.alloc_var(T.float32)  # original sigmoid (for output)
                        l_best_idx = T.alloc_var(T.int32)

                        if renormalize:
                            sel_vals = T.alloc_local([top_k], "float32")
                            sel_sum = T.alloc_var(T.float32)
                            sel_sum = T.float32(0)

                        for k in T.serial(top_k):
                            l_best_val = -T.infinity("float32")
                            l_best_orig = T.float32(0)
                            l_best_idx = T.int32(-1)
                            for j in T.serial(ELEMS_PER_THREAD):
                                if my_biased[j] > l_best_val:
                                    l_best_val = my_biased[j]
                                    l_best_orig = my_scores[j]
                                    l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                            for i in T.serial(LOG_WARP):
                                mask = T.int32(HALF_WARP) >> i
                                other_val = T.shfl_xor(l_best_val, mask)
                                other_orig = T.shfl_xor(l_best_orig, mask)
                                other_idx = T.shfl_xor(l_best_idx, mask)
                                # Must precede l_best_val/l_best_idx: it reads the old pair.
                                l_best_orig = T.if_then_else(
                                    other_val > l_best_val,
                                    other_orig,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_orig, l_best_orig
                                        ),
                                        l_best_orig,
                                    ),
                                )
                                l_best_idx = T.if_then_else(
                                    other_val > l_best_val,
                                    other_idx,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_idx, l_best_idx
                                        ),
                                        l_best_idx,
                                    ),
                                )
                                l_best_val = T.max(l_best_val, other_val)

                            if renormalize:
                                sel_vals[k] = l_best_orig
                                sel_sum = sel_sum + l_best_orig
                                if lane_id == 0:
                                    topk_ids[token_id, k] = l_best_idx
                            else:
                                if lane_id == 0:
                                    topk_weights[token_id, k] = l_best_orig
                                    topk_ids[token_id, k] = l_best_idx

                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")
                                    my_biased[j] = -T.infinity("float32")

                        if renormalize:
                            inv_sel_sum = T.alloc_var(T.float32)
                            inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        else:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], "float32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        if not renormalize:
                            inv_row_sum = T.alloc_var(T.float32)
                            inv_row_sum = T.float32(1)  # default (sigmoid / no renorm)

                        if scoring_func == "softmax":
                            l_max = T.alloc_var(T.float32)

                            l_max = -T.infinity("float32")
                            for j in T.serial(ELEMS_PER_THREAD):
                                l_max = T.max(l_max, my_scores[j])
                            for i in T.serial(LOG_WARP):
                                l_max = T.max(l_max, T.shfl_xor(l_max, T.int32(HALF_WARP) >> i))

                            if renormalize:
                                # The row sum cancels against the selected-sum divisor.
                                for j in T.serial(ELEMS_PER_THREAD):
                                    my_scores[j] = T.exp(my_scores[j] - l_max)
                            else:
                                l_sum = T.alloc_var(T.float32)

                                l_sum = T.float32(0)
                                for j in T.serial(ELEMS_PER_THREAD):
                                    val = T.exp(my_scores[j] - l_max)
                                    my_scores[j] = val
                                    l_sum = l_sum + val
                                for i in T.serial(LOG_WARP):
                                    l_sum = l_sum + T.shfl_xor(l_sum, T.int32(HALF_WARP) >> i)
                                inv_row_sum = T.float32(1) / l_sum

                        else:  # sigmoid: element-wise, no row reduction
                            for j in T.serial(ELEMS_PER_THREAD):
                                expert_idx = j * WARP_SIZE + lane_id
                                if expert_idx < num_experts:
                                    val = my_scores[j]
                                    my_scores[j] = T.float32(1) / (T.float32(1) + T.exp(-val))
                                # Padding already -inf; sigmoid(-inf)≈0, keep -inf for argmax.

                        l_best_val = T.alloc_var(T.float32)
                        l_best_idx = T.alloc_var(T.int32)

                        if renormalize:
                            sel_vals = T.alloc_local([top_k], "float32")
                            sel_sum = T.alloc_var(T.float32)
                            sel_sum = T.float32(0)

                        for k in T.serial(top_k):
                            l_best_val = -T.infinity("float32")
                            l_best_idx = T.int32(-1)
                            for j in T.serial(ELEMS_PER_THREAD):
                                if my_scores[j] > l_best_val:
                                    l_best_val = my_scores[j]
                                    l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                            for i in T.serial(LOG_WARP):
                                mask = T.int32(HALF_WARP) >> i
                                other_val = T.shfl_xor(l_best_val, mask)
                                other_idx = T.shfl_xor(l_best_idx, mask)
                                l_best_idx = T.if_then_else(
                                    other_val > l_best_val,
                                    other_idx,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_idx, l_best_idx
                                        ),
                                        l_best_idx,
                                    ),
                                )
                                l_best_val = T.max(l_best_val, other_val)

                            if renormalize:
                                sel_vals[k] = l_best_val
                                sel_sum = sel_sum + l_best_val
                                if lane_id == 0:
                                    topk_ids[token_id, k] = l_best_idx
                            else:
                                if lane_id == 0:
                                    topk_weights[token_id, k] = l_best_val * inv_row_sum
                                    topk_ids[token_id, k] = l_best_idx

                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")

                        if renormalize:
                            inv_sel_sum = T.alloc_var(T.float32)
                            inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        return main
