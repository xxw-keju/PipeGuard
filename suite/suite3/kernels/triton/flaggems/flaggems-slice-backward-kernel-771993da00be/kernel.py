# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def slice_backward_kernel(grad_output_ptr, grad_input_ptr, numel, inner, slice_len, dim_size, start, step, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    grad = tl.load(grad_output_ptr + offsets, mask=mask)
    outer_idx = offsets // (slice_len * inner)
    slice_idx = offsets // inner % slice_len
    inner_idx = offsets % inner
    dim_index = start + slice_idx * step
    input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx
    tl.store(grad_input_ptr + input_offset, grad, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return slice_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
