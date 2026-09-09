# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def reshape_and_cache_kernel(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x, k_scale, v_scale, n: tl.constexpr):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = i < n
    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i
    head_idx = i // head_size
    head_offset = i % head_size
    x_idx = head_offset // x
    x_offset = head_offset % x
    tgt_key_idx = block_idx * num_heads * (head_size // x) * block_size * x + head_idx * (head_size // x) * block_size * x + x_idx * block_size * x + block_offset * x + x_offset
    tgt_value_idx = block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + head_offset * block_size + block_offset
    tgt_key = tl.load(key + src_key_idx, mask=mask)
    tgt_value = tl.load(value + src_value_idx, mask=mask)
    tl.store(key_cache + tgt_key_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_value_idx, tgt_value, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return reshape_and_cache_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
