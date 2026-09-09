# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def symmetric_matmul_kernel(L_inv_ptr, Out_ptr, N, batch_stride, stride_row, stride_col, out_stride_row, out_stride_col, upper: tl.constexpr):
    """Compute symmetric matrix product for cholesky_inverse.

    For lower: Out = L_inv^T @ L_inv
    For upper: Out = U_inv @ U_inv^T
    Each program handles one matrix in the batch.
    """
    pid = _program_id(0)
    base_in = pid * batch_stride
    base_out = pid * N * out_stride_row
    for i in range(N):
        for j in range(N):
            if j <= i:
                acc = tl.zeros((), dtype=L_inv_ptr.dtype.element_ty)
                if upper:
                    for k in range(N):
                        u_ik = tl.load(L_inv_ptr + base_in + i * stride_row + k * stride_col)
                        u_jk = tl.load(L_inv_ptr + base_in + j * stride_row + k * stride_col)
                        acc += u_ik * u_jk
                else:
                    for k in range(N):
                        l_ki = tl.load(L_inv_ptr + base_in + k * stride_row + i * stride_col)
                        l_kj = tl.load(L_inv_ptr + base_in + k * stride_row + j * stride_col)
                        acc += l_ki * l_kj
                tl.store(Out_ptr + base_out + i * out_stride_row + j * out_stride_col, acc)
                if i != j:
                    tl.store(Out_ptr + base_out + j * out_stride_row + i * out_stride_col, acc)

def _resolve_kernel(**pipeguard_bindings):
    return symmetric_matmul_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
