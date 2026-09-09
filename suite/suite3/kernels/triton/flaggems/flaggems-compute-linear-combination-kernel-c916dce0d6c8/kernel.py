# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def _compute_linear_combination_kernel(In, Coeff, Out, M, N, K, stride_in_k, stride_in_n, stride_coeff_m, stride_coeff_k, stride_out_m, stride_out_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = _program_id(0)
    pid_n = _program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offset_k = tl.arange(0, BLOCK_K)
    m_mask = offset_m < M
    n_mask = offset_n < N
    coeff_ptrs = Coeff + offset_m * stride_coeff_m
    in_base = In + offset_n * stride_in_n
    out_ptrs = Out + offset_m * stride_out_m + offset_n * stride_out_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_off = k + offset_k
        k_mask = k_off < K
        a = tl.load(coeff_ptrs + k_off[None, :] * stride_coeff_k, mask=m_mask & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(in_base + k_off[:, None] * stride_in_k, mask=k_mask[:, None] & n_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=False)
        coeff_ptrs += BLOCK_K * stride_coeff_k
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask & n_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _compute_linear_combination_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
