# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit(do_not_specialize=['alpha', 'beta'])
def addbmm_kernel(bias_ptr, batch1_ptr, batch2_ptr, out_ptr, alpha, beta, B, M, N, K, stride_bias_m, stride_bias_n, stride_b1_b, stride_b1_m, stride_b1_k, stride_b2_b, stride_b2_k, stride_b2_n, stride_out_m, stride_out_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, IS_FP64: tl.constexpr=False):
    """Triton kernel for addbmm: out = beta * bias + alpha * sum_b(batch1[b] @ batch2[b])"""
    pid_m = _program_id(0)
    pid_n = _program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if IS_FP64:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for b in range(B):
        b1_base = batch1_ptr + b * stride_b1_b
        b2_base = batch2_ptr + b * stride_b2_b
        a_ptrs = b1_base + offs_m[:, None] * stride_b1_m + offs_k[None, :] * stride_b1_k
        b_ptrs = b2_base + offs_k[:, None] * stride_b2_k + offs_n[None, :] * stride_b2_n
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
            if IS_FP64:
                a = a.to(tl.float32)
                b_tile = b_tile.to(tl.float32)
            acc += tl.dot(a, b_tile, allow_tf32=False)
            a_ptrs += BLOCK_SIZE_K * stride_b1_k
            b_ptrs += BLOCK_SIZE_K * stride_b2_k
    bias_ptrs = bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias_tile = tl.load(bias_ptrs, mask=out_mask, other=0.0)
    out = acc * alpha + bias_tile * beta
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, out.to(bias_tile.dtype), mask=out_mask)

def _resolve_kernel(**pipeguard_bindings):
    return addbmm_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
