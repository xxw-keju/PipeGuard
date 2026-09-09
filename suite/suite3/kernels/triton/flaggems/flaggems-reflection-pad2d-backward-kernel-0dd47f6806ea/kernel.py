# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def reflection_pad2d_backward_kernel(grad_output_ptr, grad_input_ptr, in_h, in_w, out_h, out_w, out_stride_n, out_stride_c, out_stride_h, out_stride_w, in_stride_n, in_stride_c, in_stride_h, in_stride_w, num_c, pad_left, pad_right, pad_top, pad_bottom, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // num_c
    c_idx = pid_nc % num_c
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    grad_out_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    h_in_2d = h_in_offsets[:, None]
    w_in_2d = w_in_offsets[None, :]
    h_valid = h_in_offsets < in_h
    w_valid = w_in_offsets < in_w
    valid_mask = h_valid[:, None] & w_valid[None, :]
    h_out_center = h_in_2d + pad_top
    w_out_center = w_in_2d + pad_left
    center_offset = h_out_center * out_stride_h + w_out_center * out_stride_w
    center_val = tl.load(grad_out_base + center_offset, mask=valid_mask, other=0.0)
    grad_acc += center_val
    h_out_top = pad_top - h_in_2d
    top_h_valid = (h_in_2d > 0) & (h_in_2d <= pad_top)
    top_mask = top_h_valid & valid_mask
    top_offset = h_out_top * out_stride_h + w_out_center * out_stride_w
    top_val = tl.load(grad_out_base + top_offset, mask=top_mask, other=0.0)
    grad_acc += top_val
    h_out_bottom = 2 * in_h - 2 - h_in_2d + pad_top
    bottom_h_valid = (h_in_2d >= in_h - 1 - pad_bottom) & (h_in_2d < in_h - 1)
    bottom_mask = bottom_h_valid & valid_mask
    bottom_offset = h_out_bottom * out_stride_h + w_out_center * out_stride_w
    bottom_val = tl.load(grad_out_base + bottom_offset, mask=bottom_mask, other=0.0)
    grad_acc += bottom_val
    w_out_left = pad_left - w_in_2d
    left_w_valid = (w_in_2d > 0) & (w_in_2d <= pad_left)
    left_mask = left_w_valid & valid_mask
    left_offset = h_out_center * out_stride_h + w_out_left * out_stride_w
    left_val = tl.load(grad_out_base + left_offset, mask=left_mask, other=0.0)
    grad_acc += left_val
    w_out_right = 2 * in_w - 2 - w_in_2d + pad_left
    right_w_valid = (w_in_2d >= in_w - 1 - pad_right) & (w_in_2d < in_w - 1)
    right_mask = right_w_valid & valid_mask
    right_offset = h_out_center * out_stride_h + w_out_right * out_stride_w
    right_val = tl.load(grad_out_base + right_offset, mask=right_mask, other=0.0)
    grad_acc += right_val
    tl_mask = top_h_valid & left_w_valid & valid_mask
    tl_offset = h_out_top * out_stride_h + w_out_left * out_stride_w
    tl_val = tl.load(grad_out_base + tl_offset, mask=tl_mask, other=0.0)
    grad_acc += tl_val
    tr_mask = top_h_valid & right_w_valid & valid_mask
    tr_offset = h_out_top * out_stride_h + w_out_right * out_stride_w
    tr_val = tl.load(grad_out_base + tr_offset, mask=tr_mask, other=0.0)
    grad_acc += tr_val
    bl_mask = bottom_h_valid & left_w_valid & valid_mask
    bl_offset = h_out_bottom * out_stride_h + w_out_left * out_stride_w
    bl_val = tl.load(grad_out_base + bl_offset, mask=bl_mask, other=0.0)
    grad_acc += bl_val
    br_mask = bottom_h_valid & right_w_valid & valid_mask
    br_offset = h_out_bottom * out_stride_h + w_out_right * out_stride_w
    br_val = tl.load(grad_out_base + br_offset, mask=br_mask, other=0.0)
    grad_acc += br_val
    grad_in_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    store_offset = h_in_offsets[:, None] * in_stride_h + w_in_offsets[None, :] * in_stride_w
    tl.store(grad_in_base + store_offset, grad_acc.to(grad_input_ptr.type.element_ty), mask=valid_mask)

def _resolve_kernel(**pipeguard_bindings):
    return reflection_pad2d_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
