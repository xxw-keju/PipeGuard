# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _uint_to_uniform_float(x):
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64))
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale
PI = tl.constexpr(3.141592653589793)

@triton.jit
def uniform_to_cauchy(u, median, sigma):
    u = tl.maximum(1e-07, u)
    u = tl.minimum(1.0 - 1e-07, u)
    angle = PI * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))
configs = [triton.Config({'BLOCK': 256}, num_warps=8, num_stages=2), triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2), triton.Config({'BLOCK': 512}, num_warps=8, num_stages=3), triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2), triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=3), triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=4)]

@triton.jit(do_not_specialize=['philox_seed', 'philox_offset', 'median', 'sigma'])
def cauchy_kernel(out_ptr, N, median, sigma, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 4294967295).to(tl.uint32)
    c1 = (philox_offset >> 32 & 4294967295).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = _uint_to_uniform_float(r0)
    r1 = _uint_to_uniform_float(r1)
    r2 = _uint_to_uniform_float(r2)
    r3 = _uint_to_uniform_float(r3)
    c0 = uniform_to_cauchy(r0, median, sigma)
    c1 = uniform_to_cauchy(r1, median, sigma)
    c2 = uniform_to_cauchy(r2, median, sigma)
    c3 = uniform_to_cauchy(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, c0, mask=off_0 < N, eviction_policy='evict_first')
    tl.store(out_ptr + off_1, c1, mask=off_1 < N, eviction_policy='evict_first')
    tl.store(out_ptr + off_2, c2, mask=off_2 < N, eviction_policy='evict_first')
    tl.store(out_ptr + off_3, c3, mask=off_3 < N, eviction_policy='evict_first')

def _resolve_kernel(**pipeguard_bindings):
    return cauchy_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
