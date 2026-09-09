# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit(do_not_specialize=['alpha', 'beta'])
def addmv_kernel(A, B, Inp, Out, N, M, alpha, beta, stride_an, stride_am, stride_bm, stride_in, stride_outn, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = _program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        A_block_ptrs = A + offset_n * stride_an + (m + offset_m) * stride_am
        B_block_ptrs = B + (m + offset_m) * stride_bm
        a = tl.load(A_block_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_block_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
    acc = tl.sum(acc, axis=1)[:, None]
    Inp_ptrs = Inp + offset_n * stride_in
    inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    Out_ptrs = Out + offset_n * stride_outn
    out_block = acc * alpha + inp * beta
    tl.store(Out_ptrs, out_block, mask=n_mask)

def _resolve_kernel(**pipeguard_bindings):
    return addmv_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
