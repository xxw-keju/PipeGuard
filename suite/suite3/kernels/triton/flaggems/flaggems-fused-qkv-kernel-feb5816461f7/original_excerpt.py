@triton.jit
def _fused_qkv_kernel(
    q_ptr,
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    stride_q_tok,
    stride_q_head,
    stride_kv_tok,
    stride_cache_block,
    stride_cache_token,
    stride_cos_sin_pos,
    eps,
    num_q_items,
    total_items,
    NUM_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    item_id = pid

    while item_id < total_items:
        if item_id < num_q_items:
            # ═══ Q path: RMSNorm (no weight) + GPT-J RoPE ═══
            tok_idx = item_id // NUM_HEADS
            head_idx = item_id % NUM_HEADS
            base = tok_idx * stride_q_tok + head_idx * stride_q_head

            # Load all 512 dims bf16 → fp32 (single contiguous load)
            # HEAD_DIM=512 hardcoded for DeepSeek-V4
            offs = tl.arange(0, 512)
            x = tl.load(q_ptr + base + offs).to(tl.float32)

            # RMSNorm: rsqrt(mean(x²) + eps)
            # HEAD_DIM=512 hardcoded
            sq_sum = tl.sum(x * x, axis=0)
            rsqrt_val = tl.math.rsqrt(sq_sum / 512.0 + eps)
            x_normed = x * rsqrt_val

            # GPT-J RoPE: apply rotation to normed values in-register
            pos = tl.load(position_ids_ptr + tok_idx)
            # HALF_ROPE_DIM=32 hardcoded (ROPE_DIM=64)
            half_offs = tl.arange(0, 32)
            cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
            sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

            # Build rope-applied output for last 64 dims in-register
            # NOPE_DIM=448 hardcoded
            rope_even = 448 + half_offs * 2
            rope_odd = rope_even + 1

            # Gather normed even/odd values from original input + multiply
            # by rsqrt (avoids scatter issue with x_normed)
            x_re = tl.load(q_ptr + base + rope_even).to(tl.float32) * rsqrt_val
            x_ro = tl.load(q_ptr + base + rope_odd).to(tl.float32) * rsqrt_val

            q_out_e = x_re * cos - x_ro * sin
            q_out_o = x_re * sin + x_ro * cos

            # Store: first 448 dims (normed, no RoPE) + rotated rope dims
            # NOPE_DIM=448 hardcoded
            nope_mask = offs < 448
            tl.store(q_ptr + base + offs, x_normed.to(tl.bfloat16), mask=nope_mask)
            tl.store(q_ptr + base + rope_even, q_out_e.to(tl.bfloat16))
            tl.store(q_ptr + base + rope_odd, q_out_o.to(tl.bfloat16))

        else:
            # ═══ KV path: RoPE + bf16 paged cache insert ═══
            kv_idx = item_id - num_q_items
            slot_id = tl.load(slot_mapping_ptr + kv_idx)

            if slot_id >= 0:
                kv_base = kv_idx * stride_kv_tok

                # Load all 512 dims
                # HEAD_DIM=512 hardcoded
                offs = tl.arange(0, 512)
                kv_data = tl.load(kv_ptr + kv_base + offs)

                # GPT-J RoPE on last 64 dims [448, 512)
                pos = tl.load(position_ids_ptr + kv_idx)
                # HALF_ROPE_DIM=32 hardcoded
                half_offs = tl.arange(0, 32)
                cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
                sin = tl.load(
                    cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs
                )

                # NOPE_DIM=448 hardcoded
                rope_even = 448 + half_offs * 2
                rope_odd = rope_even + 1

                x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.float32)
                x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.float32)
                out_e = x_e * cos - x_o * sin
                out_o = x_e * sin + x_o * cos

                # Paged cache offset
                block_idx = slot_id // CACHE_BLOCK_SIZE
                pos_in_block = slot_id % CACHE_BLOCK_SIZE
                cache_off = (
                    block_idx * stride_cache_block + pos_in_block * stride_cache_token
                )

                # Store NoPE part (first 448 dims)
                # NOPE_DIM=448 hardcoded
                nope_mask = offs < 448
                tl.store(k_cache_ptr + cache_off + offs, kv_data, mask=nope_mask)

                # Store RoPE part (last 64 dims rotated)
                tl.store(k_cache_ptr + cache_off + rope_even, out_e.to(tl.bfloat16))
                tl.store(k_cache_ptr + cache_off + rope_odd, out_o.to(tl.bfloat16))

        item_id += tl.num_programs(0)
