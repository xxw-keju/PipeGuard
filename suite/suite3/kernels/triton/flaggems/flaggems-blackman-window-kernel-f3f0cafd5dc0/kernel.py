# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _program_id(axis: tl.constexpr):
    return tl.program_id(axis).to(tl.int64)

@triton.jit
def blackman_window_kernel(out_ptr, window_length, scale, LOW_PRECISION: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = _program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < window_length
    dtype = out_ptr.dtype.element_ty
    if LOW_PRECISION:
        n = idx.to(dtype).to(tl.float32)
        x = (n * scale).to(dtype).to(tl.float32)
        c2 = tl.cos((x * 2.0).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        c4 = tl.cos((x * 4.0).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        t4 = (c4 * 0.08).to(dtype).to(tl.float32)
        t2 = (c2 * 0.5).to(dtype).to(tl.float32)
        term = (t4 - t2).to(dtype).to(tl.float32)
        val = (term + 0.42).to(dtype)
    else:
        n = idx.to(dtype)
        x = n * scale
        val = 0.08 * tl.cos(x * 4.0) - 0.5 * tl.cos(x * 2.0) + 0.42
        val = val.to(dtype)
    tl.store(out_ptr + idx, val, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return blackman_window_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
