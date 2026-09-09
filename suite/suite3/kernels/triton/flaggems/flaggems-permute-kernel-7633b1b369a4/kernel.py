# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def permute_kernel(x_ptr, y_ptr, n_elements, ndim, in_strides_perm_ptr, out_shape_ptr, out_postfix_ptr, BLOCK_SIZE: tl.constexpr, MAX_DIMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    offs64 = offs.to(tl.int64)
    in_index = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for k in range(MAX_DIMS):
        cond = k < ndim
        step_k = tl.load(out_postfix_ptr + k, mask=cond, other=1).to(tl.int64)
        size_k = tl.load(out_shape_ptr + k, mask=cond, other=1).to(tl.int64)
        stride_k = tl.load(in_strides_perm_ptr + k, mask=cond, other=0).to(tl.int64)
        coord_k = offs64 // step_k % size_k
        in_index += coord_k * stride_k
    vals = tl.load(x_ptr + in_index, mask=mask)
    tl.store(y_ptr + offs64, vals, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return permute_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
