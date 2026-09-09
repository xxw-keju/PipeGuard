# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _fp8_fp4_mqa_logits_kernel(Q_ptr, K_ptr, K_scale_ptr, W_ptr, O_ptr, M, N, H: tl.constexpr, D: tl.constexpr, stride_qm, stride_qh, stride_qd, stride_kn, stride_kd, stride_om, stride_on, stride_wm, stride_wh, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_BLOCK: tl.constexpr):
    """Fused FP8 MQA logits with head-batched tiled dot products.

    Computes logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n])
                                  * weights[m, h])

    K is loaded once per (BLOCK_M, BLOCK_N) tile and reused across HEAD_BLOCK
    heads to minimize global memory traffic.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)
    m_mask = m_offs < M
    n_mask = n_offs < N
    k = tl.load(K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd, mask=n_mask[:, None] & (d_offs[None, :] < D), other=0.0)
    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)
        q = tl.load(Q_ptr + m_offs[:, None, None] * stride_qm + hb_offs[None, :, None] * stride_qh + d_offs[None, None, :] * stride_qd, mask=m_mask[:, None, None] & (hb_offs[None, :, None] < H) & (d_offs[None, None, :] < D), other=0.0)
        q_2d = tl.reshape(q, [BLOCK_M * HEAD_BLOCK, D])
        dot = tl.dot(q_2d, tl.trans(k))
        dot = tl.maximum(dot * k_scale[None, :], 0.0)
        w = tl.load(W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh, mask=m_mask[:, None] & (hb_offs[None, :] < H), other=0.0)
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]
        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        dot_reduced = tl.sum(dot_3d, axis=1)
        acc += dot_reduced
    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _fp8_fp4_mqa_logits_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
