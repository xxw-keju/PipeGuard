# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_lexical_alloc_scope.py#L163-L178
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def func(
        A: T.Tensor((64, 16), T.bfloat16),
        B: T.Tensor((64, 16), T.bfloat16),
        C: T.Tensor((64, 64), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.bfloat16)
            B_shared = T.alloc_shared((64, 16), T.bfloat16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for _ in T.serial(2):
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0:64, 0:64])
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
