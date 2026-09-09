# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_make_packed_api.py#L272-L275
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T, ir as I

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
