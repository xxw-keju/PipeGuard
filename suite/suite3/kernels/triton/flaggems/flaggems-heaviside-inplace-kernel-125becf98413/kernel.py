# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _heaviside_inplace_kernel(x_ptr, v_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    is_zero = x == 0
    is_neg = x < 0
    is_pos = x > 0
    res = tl.where(is_zero, v, tl.where(is_neg, 0, tl.where(is_pos, 1, x + x)))
    tl.store(x_ptr + offsets, res, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return _heaviside_inplace_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
