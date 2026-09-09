# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_reducer_v2.py#L1035-L1048
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            result = T.alloc_fragment((1,), T.float32)
            for _ in T.serial(4):
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
