# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _fused_topk_kernel(gating_ptr, topk_weights_ptr, topk_indices_ptr, token_expert_indices_ptr, e_score_correction_bias_ptr, num_tokens, num_experts: tl.constexpr, topk: tl.constexpr, renormalize: tl.constexpr, routed_scaling_factor, HAS_BIAS: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts
    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(tl.float32)
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)
    if HAS_BIAS:
        bias = tl.load(e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float('inf'))
    out_base = pid * topk
    weight_sum = 0.0
    for k_idx in tl.static_range(topk):
        max_score = tl.max(scores, axis=0)
        is_max = scores == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)
        if HAS_BIAS:
            bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
            w = max_score - bias_at_eidx
        else:
            w = max_score
        weight_sum += w
        tl.store(topk_weights_ptr + out_base + k_idx, w)
        tl.store(topk_indices_ptr + out_base + k_idx, eidx)
        tl.store(token_expert_indices_ptr + out_base + k_idx, (pid * topk + k_idx).to(tl.int32))
        scores = tl.where(expert_offsets == eidx, -float('inf'), scores)
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor
    for k_idx in tl.static_range(topk):
        w = tl.load(topk_weights_ptr + out_base + k_idx)
        tl.store(topk_weights_ptr + out_base + k_idx, w * scale)

def _resolve_kernel(**pipeguard_bindings):
    return _fused_topk_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
