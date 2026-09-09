# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_if_stmt_binding.py#L123-L128
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(check_well_formed=False)
    def expected(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            ptr = T.bind(A.data, type_annotation=A.data)
            B = T.Buffer((4,), "float32", data=ptr)
            C[0] = B[T.int32(1)]
    return expected


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
