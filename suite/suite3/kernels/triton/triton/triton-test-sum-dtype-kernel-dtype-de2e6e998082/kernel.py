# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L3100-L3104
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel_dtype(out_ptr, init, in_dtype: tl.constexpr, out_dtype: tl.constexpr):
        x = tl.full((32, 32), init, dtype=in_dtype)
        x = tl.sum(x, dtype=out_dtype)
        tl.store(out_ptr, x.to(tl.int32))
    return kernel_dtype


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
