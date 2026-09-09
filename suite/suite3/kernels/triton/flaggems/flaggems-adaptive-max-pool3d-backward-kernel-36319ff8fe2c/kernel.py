# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def adaptive_max_pool3d_backward_kernel(grad_output_ptr, indices_ptr, grad_input_ptr, in_n, in_c, in_d, in_h, in_w, out_d, out_h, out_w, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    out_per_nc = out_d * out_h * out_w
    nc_out = offsets // out_per_nc
    remainder = offsets % out_per_nc
    d_out = remainder // (out_h * out_w)
    remainder2 = remainder % (out_h * out_w)
    h_out = remainder2 // out_w
    w_out = remainder2 % out_w
    n_idx = nc_out // in_c
    c_idx = nc_out % in_c
    out_offset = n_idx * in_c * out_per_nc + c_idx * out_per_nc + d_out * out_h * out_w + h_out * out_w + w_out
    idx = tl.load(indices_ptr + out_offset, mask=mask, other=0)
    spatial_size = in_h * in_w
    d_in = idx // spatial_size
    remainder = idx % spatial_size
    h_in = remainder // in_w
    w_in = remainder % in_w
    grad_val = tl.load(grad_output_ptr + out_offset, mask=mask, other=0.0)
    in_offset = n_idx * in_c * in_d * in_h * in_w + c_idx * in_d * in_h * in_w + d_in * in_h * in_w + h_in * in_w + w_in
    tl.atomic_add(grad_input_ptr + in_offset, grad_val, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return adaptive_max_pool3d_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
