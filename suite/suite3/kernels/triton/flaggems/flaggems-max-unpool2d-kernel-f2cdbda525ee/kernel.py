# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def max_unpool2d_kernel(pooled_ptr, indices_ptr, output_ptr, pooled_stride_n, pooled_stride_c, pooled_stride_h, pooled_stride_w, out_stride_n, out_stride_c, out_stride_h, out_stride_w, n, c, pooled_h, pooled_w, out_h, out_w, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(pooled_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // c
    c_idx = pid_nc % c
    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    pooled_base_ptr = pooled_ptr + n_idx * pooled_stride_n + c_idx * pooled_stride_c
    indices_base_ptr = indices_ptr + n_idx * pooled_stride_n + c_idx * pooled_stride_c
    output_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    pooled_offsets = h_offsets[:, None] * pooled_stride_h + w_offsets[None, :] * pooled_stride_w
    indices_offsets = h_offsets[:, None] * pooled_stride_h + w_offsets[None, :] * pooled_stride_w
    pooled_mask = (h_offsets[:, None] < pooled_h) & (w_offsets[None, :] < pooled_w)
    indices_mask = pooled_mask
    pooled_vals = tl.load(pooled_base_ptr + pooled_offsets, mask=pooled_mask, other=0.0)
    indices_flat = tl.load(indices_base_ptr + indices_offsets, mask=indices_mask, other=0)
    h_orig = indices_flat // out_w
    w_orig = indices_flat % out_w
    out_offsets = h_orig * out_stride_h + w_orig * out_stride_w
    in_bounds = (h_orig >= 0) & (h_orig < out_h) & (w_orig >= 0) & (w_orig < out_w)
    out_mask = (h_offsets[:, None] < pooled_h) & (w_offsets[None, :] < pooled_w) & in_bounds
    tl.store(output_base_ptr + out_offsets, pooled_vals, mask=out_mask)

def _resolve_kernel(**pipeguard_bindings):
    return max_unpool2d_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
