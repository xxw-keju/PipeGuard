# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_nvf4_mma_block_scale.py#L713-L724
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float4_e2m1fn),
        B: T.Tensor((16,), T.float4_e2m1fn),
    ):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((3,), T.float4_e2m1fn)
            b = T.alloc_shared((4,), T.float4_e2m1fn)
            a[0] = A[0]
            b[0] = A[1]
            B[0] = a[0]
            B[1] = b[0]
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
