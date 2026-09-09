# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_negative_index.py#L7-L10
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

@T.prim_func
def negative_index_before(A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    B[0] = A[T.int32(-1)]


def _resolve_kernel(**pipeguard_bindings):
    return negative_index_before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
