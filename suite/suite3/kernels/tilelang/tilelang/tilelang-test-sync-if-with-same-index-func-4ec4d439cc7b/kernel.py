# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L134-L150
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(check_well_formed=False)
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.decl_buffer([2], dtype="float32", data=p0_arg.data)
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.alloc_buffer([1], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 8)
        T.launch_thread(threadIdx_x, 4)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if threadIdx_y < 8:
            temp_shared[threadIdx_x] = p0[0]
            temp_shared[threadIdx_x] = temp_shared[threadIdx_x]
        result_local[0] = result_local[0] + temp_shared[0]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
