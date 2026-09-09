# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_unroll_loop.py#L108-L114
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import tvm

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Buffer((4,), "int32")):
        for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
            for j in T.serial(2):
                A[i * 2 + j] = j
                if A[i * 2 + j] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
