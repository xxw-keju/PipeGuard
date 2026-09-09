# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def _batch_offsets_kernel(batch_sizes_ptr, offsets_ptr, bs32_ptr, num_steps, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < num_steps
    vals = tl.load(batch_sizes_ptr + offs, mask=mask, other=0).to(tl.int32)
    exclusive = tl.cumsum(vals, axis=0) - vals
    tl.store(offsets_ptr + offs, exclusive, mask=mask)
    tl.store(bs32_ptr + offs, vals, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return _batch_offsets_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
