# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def trace_kernel(inp_ptr, out_ptr, num_diag, stride0, stride1, BLOCK_SIZE: tl.constexpr):
    inp_dtype = inp_ptr.type.element_ty
    if inp_dtype.is_int():
        acc_dtype = tl.int64
        other_val = 0
    elif inp_dtype == tl.float64:
        acc_dtype = tl.float64
        other_val = 0.0
    else:
        acc_dtype = tl.float32
        other_val = 0.0
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    diag_stride = stride0 + stride1
    for i in range(0, tl.cdiv(num_diag, BLOCK_SIZE)):
        block_start = i * BLOCK_SIZE
        current_indices = block_start + tl.arange(0, BLOCK_SIZE)
        mask = current_indices < num_diag
        ptr_offsets = current_indices * diag_stride
        current_ptrs = inp_ptr + ptr_offsets
        vals = tl.load(current_ptrs, mask=mask, other=other_val)
        acc += vals.to(acc_dtype)
    final_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr, final_sum.to(out_ptr.type.element_ty))

def _resolve_kernel(**pipeguard_bindings):
    return trace_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
