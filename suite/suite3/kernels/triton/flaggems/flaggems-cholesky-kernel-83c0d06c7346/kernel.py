# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def cholesky_kernel(A, L, N, batch_stride, stride_a, stride_l):
    """Cholesky decomposition kernel.

    Each program computes one matrix in the batch.
    """
    pid = _program_id(0)
    a_offset = pid * batch_stride
    l_offset = pid * batch_stride
    for i in range(N):
        for j in range(i + 1):
            sum_val = tl.zeros((), dtype=A.dtype.element_ty)
            if j > 0:
                for k in range(j):
                    sum_val = sum_val + tl.load(L + l_offset + i * stride_l + k) * tl.load(L + l_offset + j * stride_l + k)
            if j == i:
                a_diag = tl.load(A + a_offset + i * stride_a + i)
                result = tl.sqrt(a_diag - sum_val)
                tl.store(L + l_offset + i * stride_l + i, result)
            else:
                a_val = tl.load(A + a_offset + i * stride_a + j)
                l_diag = tl.load(L + l_offset + j * stride_l + j)
                result = (a_val - sum_val) / l_diag
                tl.store(L + l_offset + i * stride_l + j, result)

def _resolve_kernel(**pipeguard_bindings):
    return cholesky_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
