# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def avg_pool2d_backward_kernel(grad_output_ptr, grad_input_ptr, in_c, in_h, in_w, out_h, out_w, in_stride_n, in_stride_c, in_stride_h, in_stride_w, out_stride_n, out_stride_c, out_stride_h, out_stride_w, kernel_h: tl.constexpr, kernel_w: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, padding_h: tl.constexpr, padding_w: tl.constexpr, dilation_h: tl.constexpr, dilation_w: tl.constexpr, COUNT_INCLUDE_PAD: tl.constexpr, divisor_override, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c
    grad_input_block_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_output_base_ptr = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for kh_loop in range(kernel_h):
        for kw_loop in range(kernel_w):
            h_out_num = h_in_offsets[:, None] + padding_h - kh_loop * dilation_h
            w_out_num = w_in_offsets[None, :] + padding_w - kw_loop * dilation_w
            h_valid_map = (h_out_num >= 0) & (h_out_num % stride_h == 0)
            w_valid_map = (w_out_num >= 0) & (w_out_num % stride_w == 0)
            h_out = h_out_num // stride_h
            w_out = w_out_num // stride_w
            h_out_mask = h_valid_map & (h_out < out_h)
            w_out_mask = w_valid_map & (w_out < out_w)
            out_mask = h_out_mask & w_out_mask
            if divisor_override != 0:
                divisor = tl.full((BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32)
            elif COUNT_INCLUDE_PAD:
                divisor = tl.full((BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32)
            else:
                h_start = h_out * stride_h - padding_h
                w_start = w_out * stride_w - padding_w
                count = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
                for kh_count in range(0, kernel_h):
                    for kw_count in range(0, kernel_w):
                        h_in_for_count = h_start + kh_count * dilation_h
                        w_in_for_count = w_start + kw_count * dilation_w
                        is_valid = (h_in_for_count >= 0) & (h_in_for_count < in_h) & (w_in_for_count >= 0) & (w_in_for_count < in_w)
                        count += is_valid.to(tl.int32)
                divisor = count.to(tl.float32)
            divisor = tl.where(divisor == 0, 1.0, divisor)
            grad_out_ptr = grad_output_base_ptr + h_out * out_stride_h + w_out * out_stride_w
            grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
            grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)
    grad_input_store_ptr = grad_input_block_ptr + h_in_offsets[:, None] * in_stride_h + w_in_offsets[None, :] * in_stride_w
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_store_ptr, grad_acc.to(grad_input_ptr.type.element_ty), mask=in_write_mask)

def _resolve_kernel(**pipeguard_bindings):
    return avg_pool2d_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
