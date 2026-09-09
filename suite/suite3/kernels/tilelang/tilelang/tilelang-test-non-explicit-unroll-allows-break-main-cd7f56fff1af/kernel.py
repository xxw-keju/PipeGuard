# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_unroll_loop.py#L93-L98
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import tvm

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Buffer((2,), "int32")):
        for i in T.unroll(2):
            A[i] = i
            if A[i] != 0:
                T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
