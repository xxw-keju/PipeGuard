# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L234-L251
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B_1 = T.alloc_buffer((24,), "float32", scope="shared")
        C_1 = T.alloc_buffer((1,), "float32", scope="local")
        D_1 = T.alloc_buffer((16,), "float32", scope="shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        B_1_1 = T.decl_buffer((24,), data=B_1.data, scope="shared")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1_1 = T.decl_buffer((1,), data=C_1.data, scope="local")
        C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1_1 = T.decl_buffer((16,), data=D_1.data, scope="shared")
        D_1_1[threadIdx_x] = C_1_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1_1[threadIdx_x]
    return expected


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
