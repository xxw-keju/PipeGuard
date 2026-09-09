# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def triangular_inverse_kernel(L_ptr, L_inv_ptr, N, batch_stride, stride_row, stride_col, upper: tl.constexpr):
    """Compute the inverse of a triangular matrix in-place per batch element.

    Each program handles one matrix in the batch.
    For lower triangular: forward substitution column by column.
    For upper triangular: back substitution column by column.
    """
    pid = _program_id(0)
    base = pid * batch_stride
    if upper:
        for j_rev in range(N):
            j = N - 1 - j_rev
            u_jj = tl.load(L_ptr + base + j * stride_row + j * stride_col)
            inv_jj = 1.0 / u_jj
            tl.store(L_inv_ptr + base + j * stride_row + j * stride_col, inv_jj)
            for i_off in range(N):
                i = j - 1 - i_off
                if i >= 0:
                    acc = tl.zeros((), dtype=L_ptr.dtype.element_ty)
                    for k in range(N):
                        if k > i and k <= j:
                            u_ik = tl.load(L_ptr + base + i * stride_row + k * stride_col)
                            inv_kj = tl.load(L_inv_ptr + base + k * stride_row + j * stride_col)
                            acc += u_ik * inv_kj
                    u_ii = tl.load(L_ptr + base + i * stride_row + i * stride_col)
                    result = -acc / u_ii
                    tl.store(L_inv_ptr + base + i * stride_row + j * stride_col, result)
    else:
        for j in range(N):
            l_jj = tl.load(L_ptr + base + j * stride_row + j * stride_col)
            inv_jj = 1.0 / l_jj
            tl.store(L_inv_ptr + base + j * stride_row + j * stride_col, inv_jj)
            for i in range(N):
                if i > j:
                    acc = tl.zeros((), dtype=L_ptr.dtype.element_ty)
                    for k in range(N):
                        if k >= j and k < i:
                            l_ik = tl.load(L_ptr + base + i * stride_row + k * stride_col)
                            inv_kj = tl.load(L_inv_ptr + base + k * stride_row + j * stride_col)
                            acc += l_ik * inv_kj
                    l_ii = tl.load(L_ptr + base + i * stride_row + i * stride_col)
                    result = -acc / l_ii
                    tl.store(L_inv_ptr + base + i * stride_row + j * stride_col, result)

def _resolve_kernel(**pipeguard_bindings):
    return triangular_inverse_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
