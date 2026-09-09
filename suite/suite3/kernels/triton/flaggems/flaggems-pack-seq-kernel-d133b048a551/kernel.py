# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(x_ptr, out_ptr, lengths_ptr, N: tl.constexpr, D: tl.constexpr, Lmax: tl.constexpr, PAD_VALUE: tl.constexpr, PAD_IS_UINT8: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    off_b = tl.arange(0, BLOCK_B)
    prev_lengths = tl.load(lengths_ptr + off_b, mask=off_b < pid_b, other=0)
    in_start = tl.sum(prev_lengths, axis=0)
    seq_len = tl.load(lengths_ptr + pid_b)
    t_mask = off_t < Lmax
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

def _resolve_kernel(**pipeguard_bindings):
    return _pack_seq_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
