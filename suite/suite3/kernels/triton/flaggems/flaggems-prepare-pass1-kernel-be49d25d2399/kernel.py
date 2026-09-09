# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _prepare_pass1_kernel(num_m_blocks_ptr, num_n_blocks_ptr, total_blocks_ptr, seqlen_k_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr, cu_seqlens_k_new_ptr, seqused_q_ptr, seqused_k_ptr, leftpad_k_ptr, batch, qhead_per_khead, max_seqlen_q: tl.constexpr, max_seqlen_k_new: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, HAS_CU_SEQLENS_Q: tl.constexpr, HAS_CU_SEQLENS_K: tl.constexpr, HAS_SEQUSED_Q: tl.constexpr, HAS_SEQUSED_K: tl.constexpr, HAS_LEFT_PAD: tl.constexpr, HAS_K_NEW: tl.constexpr, HAS_CU_SEQLENS_K_NEW: tl.constexpr):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_SIZE_B
    b_offs = b_start + tl.arange(0, BLOCK_SIZE_B)
    mask = b_offs < batch
    if HAS_SEQUSED_Q:
        q_len = tl.load(seqused_q_ptr + b_offs, mask=mask, other=0)
    elif HAS_CU_SEQLENS_Q:
        cur = tl.load(cu_seqlens_q_ptr + b_offs, mask=mask, other=0)
        nxt = tl.load(cu_seqlens_q_ptr + b_offs + 1, mask=mask, other=0)
        q_len = nxt - cur
    else:
        q_len = tl.full([BLOCK_SIZE_B], max_seqlen_q, dtype=tl.int32)
    q_len = q_len * qhead_per_khead
    m_blocks = (q_len + BLOCK_M - 1) // BLOCK_M
    if HAS_SEQUSED_K:
        k_len = tl.load(seqused_k_ptr + b_offs, mask=mask, other=0)
    elif HAS_CU_SEQLENS_K:
        cur = tl.load(cu_seqlens_k_ptr + b_offs, mask=mask, other=0)
        nxt = tl.load(cu_seqlens_k_ptr + b_offs + 1, mask=mask, other=0)
        k_len = nxt - cur
    else:
        k_len = tl.load(seqlen_k_ptr + b_offs, mask=mask, other=0)
    left = tl.load(leftpad_k_ptr + b_offs, mask=mask, other=0) if HAS_LEFT_PAD else 0
    if HAS_K_NEW:
        if HAS_CU_SEQLENS_K_NEW:
            cur_new = tl.load(cu_seqlens_k_new_ptr + b_offs, mask=mask, other=0)
            nxt_new = tl.load(cu_seqlens_k_new_ptr + b_offs + 1, mask=mask, other=0)
            k_len += nxt_new - cur_new
        else:
            k_len += max_seqlen_k_new
    k_len = k_len - left
    n_blocks = (k_len + BLOCK_N - 1) // BLOCK_N
    tl.store(num_m_blocks_ptr + b_offs, m_blocks, mask=mask)
    tl.store(num_n_blocks_ptr + b_offs, n_blocks, mask=mask)
    total = m_blocks * n_blocks
    tl.atomic_add(total_blocks_ptr, tl.sum(total, axis=0))

def _resolve_kernel(**pipeguard_bindings):
    return _prepare_pass1_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
