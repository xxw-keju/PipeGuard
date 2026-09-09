# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_negative_index.py#L33-L37
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

@T.prim_func
def negative_index_symbolic_before(shift: T.int32, A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(16):
        B[i] = A[shift + i]


def _resolve_kernel(**pipeguard_bindings):
    return negative_index_symbolic_before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
