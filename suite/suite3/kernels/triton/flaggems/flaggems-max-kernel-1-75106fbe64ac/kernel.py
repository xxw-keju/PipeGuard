# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

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
def max_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = _program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)

def _resolve_kernel(**pipeguard_bindings):
    return max_kernel_1

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
