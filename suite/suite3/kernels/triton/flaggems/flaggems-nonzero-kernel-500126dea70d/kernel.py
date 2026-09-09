# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def nonzero_kernel(inp, prefix_sum, out, n_elements, shape, ndim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = _program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_vals = tl.load(inp + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1
    nonzero_mask = mask and inp_vals
    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)

def _resolve_kernel(**pipeguard_bindings):
    return nonzero_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
