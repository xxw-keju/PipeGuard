# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _paged_mqa_logits_metadata_kernel(context_lens_ptr, context_lens_stride, schedule_metadata_ptr, batch_size, split_kv, num_sms, BLOCK_SIZE: tl.constexpr):
    sm_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    ctx_lens = tl.load(context_lens_ptr + offsets * context_lens_stride, mask=mask, other=0)
    num_segs = (ctx_lens + split_kv - 1) // split_kv
    num_segs = tl.where(mask, num_segs, 0)
    prefix_sum = tl.cumsum(num_segs, axis=0)
    total_segs = tl.max(prefix_sum)
    q = total_segs // num_sms
    r = total_segs % num_sms
    min_r = sm_idx if sm_idx < r else r
    seg_starts = sm_idx * q + min_r
    is_le = (prefix_sum <= seg_starts) & mask
    q_idx = tl.sum(tl.where(is_le, 1, 0))
    prev_mask = offsets < q_idx
    prev_prefix = tl.max(tl.where(prev_mask, prefix_sum, 0))
    kv_split_idx = seg_starts - prev_prefix
    out_idx = sm_idx * 2
    tl.store(schedule_metadata_ptr + out_idx, q_idx)
    tl.store(schedule_metadata_ptr + out_idx + 1, kv_split_idx)

def _resolve_kernel(**pipeguard_bindings):
    return _paged_mqa_logits_metadata_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
