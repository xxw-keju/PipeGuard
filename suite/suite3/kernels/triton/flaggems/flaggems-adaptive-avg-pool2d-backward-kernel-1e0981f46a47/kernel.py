# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_backward_kernel(grad_output_ptr, grad_input_ptr, in_c, in_h, in_w, out_h, out_w, out_stride_n, out_stride_c, out_stride_h, out_stride_w, in_stride_n, in_stride_c, in_stride_h, in_stride_w, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """
    Backward kernel for adaptive average pooling 2D.

    For each input position (ih, iw), we need to find all output positions (oh, ow)
    whose pooling window includes (ih, iw), and accumulate:
        grad_input[ih, iw] += grad_output[oh, ow] / window_size

    The adaptive pooling window boundaries are computed as:
        start_h(oh) = (oh * in_h) // out_h
        end_h(oh) = ((oh + 1) * in_h + out_h - 1) // out_h
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    grad_output_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    grad_input_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    in_h_i64 = tl.full((), in_h, dtype=tl.int64)
    in_w_i64 = tl.full((), in_w, dtype=tl.int64)
    out_h_i64 = tl.full((), out_h, dtype=tl.int64)
    out_w_i64 = tl.full((), out_w, dtype=tl.int64)
    h_in_i64 = h_in_offsets.to(tl.int64)
    w_in_i64 = w_in_offsets.to(tl.int64)
    for oh in range(out_h):
        oh_i64 = tl.full((), oh, dtype=tl.int64)
        start_h = oh_i64 * in_h_i64 // out_h_i64
        end_h = ((oh_i64 + 1) * in_h_i64 + out_h_i64 - 1) // out_h_i64
        h_in_window = (h_in_i64[:, None] >= start_h) & (h_in_i64[:, None] < end_h)
        for ow in range(out_w):
            ow_i64 = tl.full((), ow, dtype=tl.int64)
            start_w = ow_i64 * in_w_i64 // out_w_i64
            end_w = ((ow_i64 + 1) * in_w_i64 + out_w_i64 - 1) // out_w_i64
            w_in_window = (w_in_i64[None, :] >= start_w) & (w_in_i64[None, :] < end_w)
            in_window = h_in_window & w_in_window
            window_h = end_h - start_h
            window_w = end_w - start_w
            window_size = (window_h * window_w).to(tl.float32)
            grad_out_ptr = grad_output_base + oh * out_stride_h + ow * out_stride_w
            grad_out_val = tl.load(grad_out_ptr)
            grad_out_val = grad_out_val.to(tl.float32)
            grad_contribution = grad_out_val / window_size
            grad_acc += tl.where(in_window, grad_contribution, 0.0)
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    grad_input_store_ptr = grad_input_base + h_in_offsets[:, None] * in_stride_h + w_in_offsets[None, :] * in_stride_w
    tl.store(grad_input_store_ptr, grad_acc.to(grad_input_ptr.type.element_ty), mask=in_write_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _adaptive_avg_pool2d_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
