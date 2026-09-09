# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def conv3d_forward_kernel(input_pointer, weight_pointer, output_pointer, bias_pointer, in_n, input_depth, input_height, input_width, out_c, out_depth, out_height, out_width, input_n_stride, input_c_stride, input_depth_stride, input_height_stride, input_width_stride, weight_n_stride, weight_c_stride, weight_depth_stride, weight_height_stride, weight_width_stride, output_n_stride, output_c_stride, output_depth_stride, output_height_stride, output_width_stride, weight_c: tl.constexpr, weight_depth: tl.constexpr, weight_height: tl.constexpr, weight_width: tl.constexpr, stride_depth: tl.constexpr, stride_height: tl.constexpr, stride_width: tl.constexpr, padding_depth: tl.constexpr, padding_height: tl.constexpr, padding_width: tl.constexpr, dilation_depth: tl.constexpr, dilation_height: tl.constexpr, dilation_width: tl.constexpr, groups: tl.constexpr, BLOCK_NI_DO_HO_WO: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr):
    pid_ni_do_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)
    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(0, BLOCK_NI_DO_HO_WO)
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c)[:, None]
    weight_pointer += (weight_n_stride * output_c_offset + weight_n_stride * pid_group * out_per_group_c)[None, :]
    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
        c = dhwc % BLOCK_CI_COUNT * BLOCK_CI
        dhw = dhwc // BLOCK_CI_COUNT
        dh = dhw // weight_width
        d = dh // weight_height
        h = dh % weight_height
        w = dhw % weight_width
        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_depth_offset = d * dilation_depth - padding_depth + stride_depth * output_depth_point_value
        input_height_offset = h * dilation_height - padding_height + stride_height * output_height_point_value
        input_width_offset = w * dilation_width - padding_width + stride_width * output_width_point_value
        curr_input_pointer = input_pointer + (input_c_stride * input_c_offset)[None, :] + (input_depth_stride * input_depth_offset)[:, None] + (input_height_stride * input_height_offset)[:, None] + (input_width_stride * input_width_offset)[:, None]
        curr_weight_pointer = weight_pointer + (weight_c_stride * input_c_offset)[:, None] + weight_depth_stride * d + weight_height_stride * h + weight_width_stride * w
        input_mask = (in_n_point_value < in_n)[:, None] & (input_c_offset < weight_c)[None, :] & (0 <= input_depth_offset)[:, None] & (input_depth_offset < input_depth)[:, None] & (0 <= input_height_offset)[:, None] & (input_height_offset < input_height)[:, None] & (0 <= input_width_offset)[:, None] & (input_width_offset < input_width)[:, None]
        weight_mask = (input_c_offset < weight_c)[:, None] & (output_c_offset < out_per_group_c)[None, :]
        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)
        accum += tl.dot(input_block, weight_block, allow_tf32=False)
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[None, :]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
    accum += bias
    output_pointer += (output_n_stride * in_n_point_value)[:, None] + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :] + (output_depth_stride * output_depth_point_value)[:, None] + (output_height_stride * output_height_point_value)[:, None] + (output_width_stride * output_width_point_value)[:, None]
    output_mask = (in_n_point_value < in_n)[:, None] & (output_c_offset < out_per_group_c)[None, :] & (output_depth_point_value < out_depth)[:, None] & (output_height_point_value < out_height)[:, None] & (output_width_point_value < out_width)[:, None]
    tl.store(output_pointer, accum, mask=output_mask)

def _resolve_kernel(**pipeguard_bindings):
    return conv3d_forward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
