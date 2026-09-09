# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit(do_not_specialize=['numel'])
def moe_align_block_size_stage1(topk_ids_ptr, tokens_cnts_ptr, num_experts: tl.constexpr, numel, tokens_per_thread: tl.constexpr, sorted_token_ids_ptr, expert_ids_ptr, numel_sorted_token_ids: tl.constexpr, numel_expert_ids: tl.constexpr, block_size_sorted: tl.constexpr, block_size_expert: tl.constexpr):
    pid = tl.program_id(0)
    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)
    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, -1, mask=mask_expert)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return moe_align_block_size_stage1

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
