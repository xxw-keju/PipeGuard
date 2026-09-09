# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl
rsqrt = tl.extra.libdevice.rsqrt

@triton.jit
def batch_norm_no_update_kernel(input_pointer, weight_pointer, bias_pointer, output_pointer, running_mean_pointer, running_var_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, output_batch_stride, output_feat_stride, output_spatial_stride, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    feat_pid = tl.program_id(axis=0)
    running_mean = tl.load(feat_pid + running_mean_pointer).to(tl.float32)
    running_var = tl.load(feat_pid + running_var_pointer).to(tl.float32)
    inv_std = rsqrt(running_var + eps)
    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0
    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim
            curr_input_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
            curr_output_pointer = output_pointer + output_feat_stride * feat_pid + output_batch_stride * batch_offset[:, None] + output_spatial_stride * spatial_offset[None, :]
            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
            output = weight * (curr_input - running_mean) * inv_std + bias
            tl.store(curr_output_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])

def _resolve_kernel(**pipeguard_bindings):
    return batch_norm_no_update_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
