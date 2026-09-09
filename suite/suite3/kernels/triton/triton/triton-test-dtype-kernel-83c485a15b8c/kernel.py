# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L7124-L7129
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(X):
        dtype_x: tl.constexpr = X.dtype.element_ty
        tl.static_assert(dtype_x == tl.int32)
        tl.static_assert(dtype_x == tl.constexpr(tl.int32))
        tl.static_assert(dtype_x == tl.int8 or (dtype_x == tl.int16 or dtype_x == tl.int32))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
