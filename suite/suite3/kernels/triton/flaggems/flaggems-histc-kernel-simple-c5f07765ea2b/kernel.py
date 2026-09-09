# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def histc_kernel_simple(inp_ptr, out_ptr, n_elements, bins, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    """
    Simple histogram kernel - each program handles one element at a time.
    """
    pid = _program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=float('nan'))
    inp_val = inp_val.to(tl.float32)
    bin_idx = tl.floor((inp_val - min_val) * bins / (max_val - min_val)).to(tl.int64)
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
    valid_mask = mask & in_range
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem='relaxed')

def _resolve_kernel(**pipeguard_bindings):
    return histc_kernel_simple

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
