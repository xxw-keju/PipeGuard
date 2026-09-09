# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_if_stmt_binding.py#L196-L232
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import tvm as tvm

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(
        flag: T.Buffer((2,), "int32"),
        counter_tvm: T.Buffer((1,), "int32"),
        counter_tl: T.Buffer((1,), "int32"),
        out: T.Buffer((4,), "int32"),
    ):
        if flag[0] != 0:
            snapshot_tvm = T.bind(counter_tvm[0])
            pos_tvm = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    counter_tvm.access_ptr("rw"),
                    1,
                )
            )
            out[0] = snapshot_tvm
            out[1] = pos_tvm
        if flag[1] != 0:
            snapshot_tl = T.bind(counter_tl[0])
            pos_tl = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    T.call_intrin(
                        "handle",
                        tvm.ir.Op.get("tl.access_ptr"),
                        counter_tl[0],
                        1,
                        3,
                    ),
                    1,
                )
            )
            out[2] = snapshot_tl
            out[3] = pos_tl
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
