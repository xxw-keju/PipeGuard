# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_var_init.py#L37-L45
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def test_var_assign(A: T.Tensor((2,), T.int32)):
        with T.Kernel(1) as _:
            a = T.alloc_var(T.int32, init=1)
            b = T.alloc_var(T.int32, init=a)  # b gets value of a
            a = 2
            d = T.alloc_var(T.int32, init=a)  # c gets new value of a
            A[0] = b
            A[1] = d
    return test_var_assign


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
