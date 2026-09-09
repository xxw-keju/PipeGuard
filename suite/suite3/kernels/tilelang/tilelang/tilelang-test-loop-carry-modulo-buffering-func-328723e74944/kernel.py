# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_thread_sync.py#L463-L477
# The exact upstream excerpt is stored in original_excerpt.py.

from tvm.script import tirx as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([2, 64], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Double buffering pattern: write to buffer[i%2], read from buffer[i%2]
            # After shift: write buffer[i%2], read buffer[(i+1)%2]
            # These are different buffers, so no conflict
            temp_shared[i % 2, tx] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[i % 2, tx]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
