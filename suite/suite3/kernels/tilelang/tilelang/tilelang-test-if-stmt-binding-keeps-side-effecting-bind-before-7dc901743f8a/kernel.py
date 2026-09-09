# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_if_stmt_binding.py#L163-L175
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import tvm as tvm

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A: T.Buffer((4,), "float32"), counter: T.Buffer((1,), "int32"), out: T.Buffer((4,), "int32")):
        if A[0] >= T.float32(0):
            pos = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    counter.access_ptr("rw"),
                    1,
                )
            )
            if pos < 4:
                out[pos] = 1
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
