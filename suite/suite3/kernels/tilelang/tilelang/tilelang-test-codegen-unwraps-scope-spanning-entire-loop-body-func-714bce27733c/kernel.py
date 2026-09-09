# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_lexical_alloc_scope.py#L384-L395
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def func(
        A: T.Tensor((128, 4), T.float32),
        B: T.Tensor((128, 4), T.float32),
    ):
        with T.Kernel(1, threads=128):
            for k in T.serial(4):
                with T.sblock():
                    T.sblock_attr({"lexical_alloc_scope": 1})
                    S = T.alloc_buffer((128,), dtype=T.float32, scope="local")
                    S[T.get_thread_binding()] = A[T.get_thread_binding(), k]
                    B[T.get_thread_binding(), k] = S[T.get_thread_binding()]
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
