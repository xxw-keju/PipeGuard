# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/kernel/test_tilelang_kernel_sm70_fragment_copy.py#L9-L31
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 16), T.float16),
        C: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 16), T.float16)
            P = T.alloc_fragment((64, 16), T.float32)
            P_half = T.alloc_fragment((64, 16), T.float16)
            C_local = T.alloc_fragment((64, 16), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)

            for i, j in T.Parallel(64, 16):
                P[i, j] = T.cast(A_shared[i, j], T.float32)

            T.copy(P, P_half)
            T.clear(C_local)
            T.gemm(P_half, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
