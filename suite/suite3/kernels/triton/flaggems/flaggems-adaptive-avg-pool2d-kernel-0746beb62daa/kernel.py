# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(input_ptr, output_ptr, in_stride_n, in_stride_c, in_stride_h, in_stride_w, in_c, in_h, in_w, out_h, out_w, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    h_start = h_out_offsets[:, None] * in_h // out_h
    h_end = 1 + ((h_out_offsets[:, None] + 1) * in_h - 1) // out_h
    w_start = w_out_offsets[None, :] * in_w // out_w
    w_end = 1 + ((w_out_offsets[None, :] + 1) * in_w - 1) // out_w
    window_size = (h_end - h_start) * (w_end - w_start)
    window_size = tl.maximum(window_size, 1)
    h_start = tl.maximum(h_start, 0)
    w_start = tl.maximum(w_start, 0)
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    max_win_h = (in_h + out_h - 1) // out_h + 1
    max_win_w = (in_w + out_w - 1) // out_w + 1
    for kh in range(64):
        h_in = h_start + kh
        h_valid = (h_in >= 0) & (h_in < h_end) & (h_in < in_h)
        h_valid_kw = kh < max_win_h
        for kw in range(64):
            w_in = w_start + kw
            w_valid = (w_in >= 0) & (w_in < w_end) & (w_in < in_w)
            w_valid_kw = kw < max_win_w
            h_valid_2d = h_valid & h_valid_kw
            w_valid_2d = w_valid & w_valid_kw
            in_mask = h_valid_2d & w_valid_2d
            h_in_2d = tl.reshape(h_in, (BLOCK_H, 1))
            w_in_2d = tl.reshape(w_in, (1, BLOCK_W))
            input_offset = h_in_2d * in_stride_h + w_in_2d * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset, mask=in_mask, other=0.0)
            sum_val = tl.where(in_mask, current_val, 0.0)
            sum_acc = sum_acc + tl.reshape(sum_val, (BLOCK_H, BLOCK_W))
    output_vals = sum_acc / window_size.to(tl.float32)
    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    out_mask = (out_h_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask)

def _resolve_kernel(**pipeguard_bindings):
    return adaptive_avg_pool2d_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
