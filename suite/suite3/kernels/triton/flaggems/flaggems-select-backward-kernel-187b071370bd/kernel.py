# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _select_backward_kernel(grad_ptr, out_ptr, total: tl.constexpr, inner_size: tl.constexpr, dim_stride: tl.constexpr, index: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    outer = offs // inner_size
    inner = offs % inner_size
    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = outer * dim_stride + index * inner_size + inner
    tl.store(out_ptr + out_offset, vals, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return _select_backward_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
