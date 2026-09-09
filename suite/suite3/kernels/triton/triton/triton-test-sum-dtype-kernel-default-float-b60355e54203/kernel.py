# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L3112-L3116
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel_default_float(out_ptr):
        x = tl.full((32, 32), 1.0, dtype=tl.bfloat16)
        x = tl.sum(x)
        tl.store(out_ptr, x)
    return kernel_default_float


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
