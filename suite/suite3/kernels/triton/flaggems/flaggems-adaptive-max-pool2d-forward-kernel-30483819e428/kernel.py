# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

def get_dtype_min(dtype):
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float('-inf')
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2 ** (width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value

@triton.jit
def adaptive_max_pool2d_forward_kernel(input_ptr, output_ptr, indices_ptr, in_stride_n, in_stride_c, in_stride_h, in_stride_w, in_c, in_h, in_w, out_h, out_w, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    h_start = h_out_offsets[:, None] * in_h // out_h
    h_end = ((h_out_offsets[:, None] + 1) * in_h + out_h - 1) // out_h
    w_start = w_out_offsets[None, :] * in_w // out_w
    w_end = ((w_out_offsets[None, :] + 1) * in_w + out_w - 1) // out_w
    max_window_h = tl.cdiv(in_h, out_h) + 1
    max_window_w = tl.cdiv(in_w, out_w) + 1
    for kh in range(0, max_window_h):
        for kw in range(0, max_window_w):
            h_in = h_start + kh
            w_in = w_start + kw
            in_window_mask = (kh < h_end - h_start) & (kw < w_end - w_start)
            in_mask = in_window_mask & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset, mask=in_mask, other=min_val)
            current_idx = h_in * in_w + w_in
            is_new_max = ((current_val > max_val_acc) | (current_val != current_val) | (max_idx_acc < 0)) & in_mask
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max, current_idx, max_idx_acc)
    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    indices_block_ptr = indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)

def _resolve_kernel(**pipeguard_bindings):
    return adaptive_max_pool2d_forward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
