# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def prelu_kernel(x_ptr, w_ptr, out_ptr, n_elements, S, C, w_is_scalar: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if w_is_scalar:
        alpha = tl.load(w_ptr)
        y = tl.where(x >= 0, x, alpha * x)
    else:
        c = offsets // S % C
        alpha = tl.load(w_ptr + c, mask=mask)
        y = tl.where(x >= 0, x, alpha * x)
    tl.store(out_ptr + offsets, y, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return prelu_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
