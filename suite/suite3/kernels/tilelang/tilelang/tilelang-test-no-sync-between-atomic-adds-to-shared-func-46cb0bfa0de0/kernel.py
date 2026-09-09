# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L33-L55
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import tvm as tvm

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16, 128), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.tvm_access_ptr(
                        T.type_annotation("float32"),
                        A_shared.data,
                        i * 128 + tx,
                        1,
                        3,
                    ),
                    T.float32(1),
                    T.int32(0),
                )
            )
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
