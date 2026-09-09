# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L67-L83
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(private=True)
    def func():
        buf_dyn_shmem = T.alloc_buffer((2048,), "uint8", scope="shared.dyn")
        x_ub: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        y_ub: T.handle("float8_e4m3fn", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        x_local = T.alloc_buffer((4,), "float32", scope="local")
        y_local = T.alloc_buffer((4,), "float8_e4m3fn", scope="local")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        x_ub_1 = T.decl_buffer((512,), "float32", data=x_ub, scope="shared.dyn")
        y_ub_1 = T.decl_buffer((512,), "float8_e4m3fn", data=y_ub, scope="shared.dyn")
        x_local_1 = T.decl_buffer((4,), "float32", data=x_local.data, scope="local")
        y_local_1 = T.decl_buffer((4,), "float8_e4m3fn", data=y_local.data, scope="local")
        x_local_1[T.Ramp(0, 1, 4)] = x_ub_1[T.Ramp(tx * 4, 1, 4)]
        y_local_1[T.Ramp(0, 1, 4)] = T.Cast("float8_e4m3fnx4", x_local_1[T.Ramp(0, 1, 4)])
        y_ub_1[T.Ramp(tx * 4, 1, 4)] = y_local_1[T.Ramp(0, 1, 4)]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
