# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_unroll_loop.py#L55-L62
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(n: T.int32, A: T.Buffer((16,), "int32")):
        for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
            A[i] = i
        for j in T.unroll(n):
            A[j] = j
        for k in T.unroll(8, annotations={"pragma_unroll_factor": 4}):
            A[k + 8] = k
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
