# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def ldl_factor_kernel(A, LD, pivots, N, stride_a0, stride_a1, stride_a2, stride_ld0, stride_ld1, stride_ld2, MAX_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    if tl.program_id(1) == 0:
        for i in range(MAX_SIZE):
            row_start = A + batch_idx * stride_a0 + i * stride_a1
            row = tl.load(row_start + tl.arange(0, MAX_SIZE) * stride_a2, mask=i < N)
            ld_row_start = LD + batch_idx * stride_ld0 + i * stride_ld1
            tl.store(ld_row_start + tl.arange(0, MAX_SIZE) * stride_ld2, row, mask=i < N)
        k = 0
        while k < N:
            d_kk = tl.load(LD + batch_idx * stride_ld0 + k * stride_ld1 + k * stride_ld2)
            j = 0
            while j < k:
                d_jj = tl.load(LD + batch_idx * stride_ld0 + j * stride_ld1 + j * stride_ld2)
                l_kj = tl.load(LD + batch_idx * stride_ld0 + k * stride_ld1 + j * stride_ld2)
                d_kk = d_kk - l_kj * l_kj * d_jj
                j = j + 1
            tl.store(LD + batch_idx * stride_ld0 + k * stride_ld1 + k * stride_ld2, d_kk)
            i_zero = 0
            while i_zero < k:
                tl.store(LD + batch_idx * stride_ld0 + i_zero * stride_ld1 + k * stride_ld2, 0.0)
                i_zero = i_zero + 1
            i = k + 1
            while i < N:
                l_ik = tl.load(LD + batch_idx * stride_ld0 + i * stride_ld1 + k * stride_ld2)
                j_inner = 0
                while j_inner < k:
                    d_jj = tl.load(LD + batch_idx * stride_ld0 + j_inner * stride_ld1 + j_inner * stride_ld2)
                    l_ij = tl.load(LD + batch_idx * stride_ld0 + i * stride_ld1 + j_inner * stride_ld2)
                    l_kj = tl.load(LD + batch_idx * stride_ld0 + k * stride_ld1 + j_inner * stride_ld2)
                    l_ik = l_ik - l_ij * l_kj * d_jj
                    j_inner = j_inner + 1
                l_ik = l_ik / d_kk
                tl.store(LD + batch_idx * stride_ld0 + i * stride_ld1 + k * stride_ld2, l_ik)
                i = i + 1
            k = k + 1
        j = 0
        while j < N:
            i = 0
            while i < j:
                tl.store(LD + batch_idx * stride_ld0 + i * stride_ld1 + j * stride_ld2, 0.0)
                i = i + 1
            j = j + 1
        for i in range(MAX_SIZE):
            pivots_ptr = pivots + batch_idx * N + i
            tl.store(pivots_ptr, i + 1, mask=i < N)

def _resolve_kernel(**pipeguard_bindings):
    return ldl_factor_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
