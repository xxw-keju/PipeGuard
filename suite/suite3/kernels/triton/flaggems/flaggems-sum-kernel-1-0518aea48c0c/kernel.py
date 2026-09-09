# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def sum_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(inp.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty
    pid = _program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)

def _resolve_kernel(**pipeguard_bindings):
    return sum_kernel_1

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
