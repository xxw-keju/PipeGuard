# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def sparse_attn_triton_kernel(Q, KV, O, attn_sink, topk_idxs, stride_qb, stride_qm, stride_qh, stride_qd, stride_kvb, stride_kvn, stride_kvd, stride_ob, stride_om, stride_oh, stride_od, stride_idxb, stride_idxm, stride_idxk, scale, topk, H_ACTUAL, BLOCK: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm
    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)
    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)
    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < topk
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1
        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)
        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * scale
        acc_s = tl.where(valid_mask[None, :], acc_s, float('-inf'))
        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)
        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)
        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum
    sink_vals = tl.load(attn_sink + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)
    acc_o = acc_o / sum_exp[:, None]
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])

def _resolve_kernel(**pipeguard_bindings):
    return sparse_attn_triton_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
