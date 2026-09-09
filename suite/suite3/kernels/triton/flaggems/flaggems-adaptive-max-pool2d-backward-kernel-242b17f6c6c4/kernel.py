# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def adaptive_max_pool2d_backward_kernel(grad_output_ptr, indices_ptr, grad_input_ptr, in_h, in_w, out_h, out_w, out_numel, in_numel_per_nc, out_stride_nc, BLOCK_SIZE: tl.constexpr):
    """
    Each program processes a block of output elements.
    For each output element, scatter its grad_output value to the position
    indicated by the corresponding index in grad_input.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_numel
    grad_out_val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    indices_val = tl.load(indices_ptr + offsets, mask=mask, other=0)
    nc_idx = offsets // out_stride_nc
    grad_input_offset = nc_idx * in_numel_per_nc + indices_val
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_out_val, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return adaptive_max_pool2d_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
