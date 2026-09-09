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
def fractional_max_pool2d_forward_kernel(input_ptr, output_ptr, indices_ptr, random_samples_ptr, in_stride_n, in_stride_c, in_stride_h, in_stride_w, in_c, in_h, in_w, out_h, out_w, alpha_h, alpha_w, kernel_h: tl.constexpr, kernel_w: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """Forward kernel for fractional max pool 2d.

    Computes pooling region start positions inline from random_samples,
    using the formula: start[i] = int((i + sample) * alpha) - int(sample * alpha).
    This eliminates precomputed h_starts/w_starts arrays and their Python overhead.
    """
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
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    sample_h = tl.load(random_samples_ptr + pid_nc * 2).to(tl.float32)
    sample_w = tl.load(random_samples_ptr + pid_nc * 2 + 1).to(tl.float32)
    h_mask = h_out_offsets < out_h
    w_mask = w_out_offsets < out_w
    sample_alpha_h = (sample_h * alpha_h).to(tl.int32)
    sample_alpha_w = (sample_w * alpha_w).to(tl.int32)
    h_starts = ((h_out_offsets.to(tl.float32) + sample_h) * alpha_h).to(tl.int32) - sample_alpha_h
    w_starts = ((w_out_offsets.to(tl.float32) + sample_w) * alpha_w).to(tl.int32) - sample_alpha_w
    h_starts = tl.where(h_out_offsets == out_h - 1, in_h - kernel_h, h_starts)
    w_starts = tl.where(w_out_offsets == out_w - 1, in_w - kernel_w, w_starts)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)
    h_base = h_starts[:, None]
    w_base = w_starts[None, :]
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_base + kh
            w_in = w_base + kw
            in_mask = h_mask[:, None] & w_mask[None, :] & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset, mask=in_mask, other=min_val)
            current_idx = h_in * in_w + w_in
            update_mask = current_val > max_val_acc
            max_val_acc = tl.where(update_mask, current_val, max_val_acc)
            max_idx_acc = tl.where(update_mask, current_idx, max_idx_acc)
    out_nc_offset = pid_nc * out_h * out_w
    out_offsets = out_nc_offset + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    out_mask = h_mask[:, None] & w_mask[None, :]
    tl.store(output_ptr + out_offsets, max_val_acc, mask=out_mask)
    tl.store(indices_ptr + out_offsets, max_idx_acc, mask=out_mask)

def _resolve_kernel(**pipeguard_bindings):
    return fractional_max_pool2d_forward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
