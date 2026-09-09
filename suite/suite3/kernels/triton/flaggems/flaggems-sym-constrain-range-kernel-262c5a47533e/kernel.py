# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def sym_constrain_range_kernel(size_ptr, min_ptr, max_ptr, out_ptr):
    size = tl.load(size_ptr)
    lo = tl.load(min_ptr)
    hi = tl.load(max_ptr)
    below = size < lo
    above = size > hi
    violated = below or above
    tl.store(out_ptr, violated.to(tl.int32))

def _resolve_kernel(**pipeguard_bindings):
    return sym_constrain_range_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
