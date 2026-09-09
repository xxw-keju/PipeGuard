    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_prefill_paged_with_kv_cache_rope_append_func(block_m: int, threads: int) -> Callable:
        kv_new_shape = (total_q, heads_kv, dim)
        kv_pages_shape = (physical_tokens, heads_kv, dim)
        block_table_shape = (batch, max_pages_per_req)
        rope_shape = (max_position, half)

        @T.prim_func
        def _gqa_prefill_paged_with_kv_cache_rope_append_main(
            k_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
            v_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
            k_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
            v_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cache_seqlens: T.Tensor([batch], T.int32),  # type: ignore
            block_table: T.Tensor(block_table_shape, T.int32),  # type: ignore
            cos_table: T.Tensor(rope_shape, dtype),  # type: ignore
            sin_table: T.Tensor(rope_shape, dtype),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(max_seqlen_q, block_m), heads_kv, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_start = cu_seqlens_q[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                old_len = cache_seqlens[bz]
                append_start = old_len + bx * block_m
                append_end = append_start + block_m

                if bx * block_m + block_m <= q_len:
                    if append_start >> T.int32(page_size_log2) == (append_end - 1) >> T.int32(
                        page_size_log2
                    ):
                        page_idx = append_start >> T.int32(page_size_log2)
                        page_offset = append_start - page_idx * page_size
                        physical_start = block_table[bz, page_idx] * page_size + page_offset
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            logical_pos = old_len + new_pos
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half, T.if_then_else(d < rotary_dim, d - half, d)
                            )
                            c = T.Cast("float32", cos_table[logical_pos, freq_idx])
                            s = T.Cast("float32", sin_table[logical_pos, freq_idx])
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = T.Cast("float32", k_new[q_start + new_pos, by, paired_d])
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_start + i, by, d] = T.if_then_else(
                                d < rotary_dim,
                                T.Cast(dtype, T.Cast("float32", val) * c + rotated * s),
                                val,
                            )
                            v_pages[physical_start + i, by, d] = v_new[q_start + new_pos, by, d]
                    else:
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            logical_pos = old_len + new_pos
                            page_idx = logical_pos >> T.int32(page_size_log2)
                            page_offset = logical_pos - page_idx * page_size
                            physical_pos = block_table[bz, page_idx] * page_size + page_offset
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half, T.if_then_else(d < rotary_dim, d - half, d)
                            )
                            c = T.Cast("float32", cos_table[logical_pos, freq_idx])
                            s = T.Cast("float32", sin_table[logical_pos, freq_idx])
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = T.Cast("float32", k_new[q_start + new_pos, by, paired_d])
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_pos, by, d] = T.if_then_else(
                                d < rotary_dim,
                                T.Cast(dtype, T.Cast("float32", val) * c + rotated * s),
                                val,
                            )
                            v_pages[physical_pos, by, d] = v_new[q_start + new_pos, by, d]
                else:
                    for i, d in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        safe_new_pos = T.if_then_else(new_pos < q_len, new_pos, 0)
                        logical_pos = old_len + safe_new_pos
                        page_idx = logical_pos >> T.int32(page_size_log2)
                        page_offset = logical_pos - page_idx * page_size
                        if new_pos < q_len:
                            physical_pos = block_table[bz, page_idx] * page_size + page_offset
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half, T.if_then_else(d < rotary_dim, d - half, d)
                            )
                            c = T.Cast("float32", cos_table[logical_pos, freq_idx])
                            s = T.Cast("float32", sin_table[logical_pos, freq_idx])
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = T.Cast("float32", k_new[q_start + new_pos, by, paired_d])
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_pos, by, d] = T.if_then_else(
                                d < rotary_dim,
                                T.Cast(dtype, T.Cast("float32", val) * c + rotated * s),
                                val,
                            )
                            v_pages[physical_pos, by, d] = v_new[q_start + new_pos, by, d]

        return _gqa_prefill_paged_with_kv_cache_rope_append_main
