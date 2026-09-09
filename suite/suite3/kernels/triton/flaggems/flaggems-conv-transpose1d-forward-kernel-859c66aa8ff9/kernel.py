# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_forward_kernel(input_pointer, weight_pointer, output_pointer, bias_pointer, batch_size, input_width, out_channels, out_width, input_n_stride, input_c_stride, input_w_stride, weight_ic_stride, weight_oc_stride, weight_w_stride, output_n_stride, output_c_stride, output_w_stride, in_channels: tl.constexpr, kernel_width: tl.constexpr, stride_width: tl.constexpr, padding_width: tl.constexpr, dilation_width: tl.constexpr, groups: tl.constexpr, BLOCK_N_OW: tl.constexpr, BLOCK_IC: tl.constexpr, BLOCK_OC: tl.constexpr):
    """
    Triton kernel for 1D transposed convolution forward pass.

    For transposed convolution:
    - input has shape (N, in_channels, in_width)
    - weight has shape (in_channels, out_channels/groups, kernel_width)
    - output has shape (N, out_channels, out_width)

    The output at position o is computed by summing contributions from all input
    positions i where the kernel at position k could have produced output at o:
    o = i * stride - padding + k * dilation
    => i = (o + padding - k * dilation) / stride (must be integer)
    """
    pid_n_ow = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)
    n_ow_offset = pid_n_ow * BLOCK_N_OW + tl.arange(0, BLOCK_N_OW)
    batch_idx = n_ow_offset // out_width
    out_w_idx = n_ow_offset % out_width
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels
    oc_offset = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    accum = tl.zeros((BLOCK_N_OW, BLOCK_OC), dtype=tl.float32)
    input_base = input_pointer + (input_n_stride * batch_idx)[:, None] + input_c_stride * pid_group * in_channels_per_group
    weight_base = weight_pointer + weight_ic_stride * pid_group * in_channels_per_group + (weight_oc_stride * oc_offset)[None, :]
    BLOCK_IC_COUNT = (in_channels_per_group + BLOCK_IC - 1) // BLOCK_IC
    for ic_k in range(BLOCK_IC_COUNT * kernel_width):
        ic_block = ic_k // kernel_width * BLOCK_IC
        k = ic_k % kernel_width
        ic_offset = ic_block + tl.arange(0, BLOCK_IC)
        numerator = out_w_idx + padding_width - k * dilation_width
        is_divisible = numerator % stride_width == 0
        in_w_idx = numerator // stride_width
        curr_input_pointer = input_base + (input_c_stride * ic_offset)[None, :] + (input_w_stride * in_w_idx)[:, None]
        input_mask = (batch_idx < batch_size)[:, None] & (ic_offset < in_channels_per_group)[None, :] & is_divisible[:, None] & (in_w_idx >= 0)[:, None] & (in_w_idx < input_width)[:, None]
        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        curr_weight_pointer = weight_base + (weight_ic_stride * ic_offset)[:, None] + weight_w_stride * k
        weight_mask = (ic_offset < in_channels_per_group)[:, None] & (oc_offset < out_channels_per_group)[None, :]
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)
        accum += tl.dot(input_block.to(tl.float32), weight_block.to(tl.float32), allow_tf32=False)
    bias_ptr = bias_pointer + pid_group * out_channels_per_group + oc_offset
    bias_mask = oc_offset < out_channels_per_group
    bias = tl.load(bias_ptr, mask=bias_mask, other=0.0).to(tl.float32)
    accum += bias[None, :]
    output_ptr = output_pointer + (output_n_stride * batch_idx)[:, None] + (output_c_stride * (pid_group * out_channels_per_group + oc_offset))[None, :] + (output_w_stride * out_w_idx)[:, None]
    output_mask = (batch_idx < batch_size)[:, None] & (oc_offset < out_channels_per_group)[None, :] & (out_w_idx < out_width)[:, None]
    tl.store(output_ptr, accum, mask=output_mask)

def _resolve_kernel(**pipeguard_bindings):
    return conv_transpose1d_forward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
