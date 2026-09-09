# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _get_cos_sin(cos_sin_cache_ptr, cos_sin_cache_stride, pos, HALF_ROT_DIM: tl.constexpr):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return (cos, sin)

@triton.jit
def _fused_indexer_q_rope_quant_kernel(pos_ptr, index_q_ptr, index_q_stride0, index_q_stride1, index_q_cos_sin_ptr, index_q_cos_sin_stride, INDEX_Q_HALF_ROT_DIM: tl.constexpr, index_q_fp8_ptr, index_q_fp8_stride0, index_q_fp8_stride1, INDEX_Q_HEAD_DIM: tl.constexpr, index_weights_ptr, index_weights_stride, index_weights_softmax_scale, index_weights_head_scale, index_weights_out_ptr, index_weights_out_stride):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(index_q_cos_sin_ptr, index_q_cos_sin_stride, pos, INDEX_Q_HALF_ROT_DIM)
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 0.0001), 448.0)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))
    fp8_base_ptr = index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(fp8_base_ptr + nope_offset, tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv))
    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
    tl.store(fp8_rot_base + half_offset * 2, tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv))
    tl.store(fp8_rot_base + half_offset * 2 + 1, tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv))
    index_weights = tl.load(index_weights_ptr + tok_idx * index_weights_stride + head_idx)
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx, index_weights)

def _resolve_kernel(**pipeguard_bindings):
    return _fused_indexer_q_rope_quant_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
