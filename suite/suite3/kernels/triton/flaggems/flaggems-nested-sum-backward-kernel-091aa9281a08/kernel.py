# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _nested_sum_backward_kernel(grad_ptr, out_ptr, numel, dim_size, inner_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    outer_idx = offsets // (dim_size * inner_size)
    inner_idx = offsets % inner_size
    grad_offset = outer_idx * inner_size + inner_idx
    grad = tl.load(grad_ptr + grad_offset, mask=mask)
    tl.store(out_ptr + offsets, grad, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return _nested_sum_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
