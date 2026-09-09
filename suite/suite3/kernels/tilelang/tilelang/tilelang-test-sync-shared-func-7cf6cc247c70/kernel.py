# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L215-L232
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(private=True)
    def func(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B = T.alloc_buffer((24,), "float32", scope="shared")
        C = T.alloc_buffer((1,), "float32", scope="local")
        D = T.alloc_buffer((16,), "float32", scope="shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1 = T.decl_buffer((24,), data=B.data, scope="shared")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1 = T.decl_buffer((1,), data=C.data, scope="local")
        C_1[0] = B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1 = T.decl_buffer((16,), data=D.data, scope="shared")
        D_1[threadIdx_x] = C_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1[threadIdx_x]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
