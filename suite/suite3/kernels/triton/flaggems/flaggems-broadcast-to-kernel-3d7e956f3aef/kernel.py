# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def broadcast_to_kernel(x_ptr, out_ptr, n_elements, ndims, out_shape_ptr, out_cumprod_ptr, in_stride_ptr, BLOCK_SIZE: tl.constexpr, MAX_DIMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for d in range(MAX_DIMS):
        s = tl.load(out_shape_ptr + d)
        stride_right = tl.load(out_cumprod_ptr + d)
        in_stride = tl.load(in_stride_ptr + d)
        idx_d = offsets // stride_right % s
        in_offsets += idx_d * in_stride
    x = tl.load(x_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return broadcast_to_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
