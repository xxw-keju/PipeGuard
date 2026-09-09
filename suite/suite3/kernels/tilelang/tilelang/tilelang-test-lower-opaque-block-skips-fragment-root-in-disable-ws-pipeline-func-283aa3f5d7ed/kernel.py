# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_lexical_alloc_scope.py#L282-L299
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def func(
        A: T.Tensor((256, 256), T.bfloat16),
        B: T.Tensor((128, 256), T.bfloat16),
        C: T.Tensor((256, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((256, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_local = T.alloc_fragment((256, 128), T.float32)
            C_shared = T.alloc_shared((256, 128), T.bfloat16)
            T.clear(C_local)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[0, k * 128], A_shared)
                T.copy(B[0, k * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[0:256, 0:128])
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
