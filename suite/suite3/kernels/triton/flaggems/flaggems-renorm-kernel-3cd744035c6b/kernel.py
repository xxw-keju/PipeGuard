# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit(do_not_specialize=['p', 'maxnorm'])
def renorm_kernel(X, N, p, maxnorm, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + pid * N + offset, mask=mask, other=0.0).to(tl.float32)
    _sum = tl.sum(tl.extra.libdevice.pow(tl.abs(x), p))
    norm = tl.extra.libdevice.pow(_sum, 1.0 / p)
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)
    x = x * scale
    tl.store(X + pid * N + offset, x, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return renorm_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
