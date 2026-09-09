@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values, mask=mask)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)
