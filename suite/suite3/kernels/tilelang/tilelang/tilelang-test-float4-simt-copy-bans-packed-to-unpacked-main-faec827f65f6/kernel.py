# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_float4_e2m1_unpacked_dtype.py#L62-L68
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((64, 128), T.float4_e2m1fn),
    ):
        with T.Kernel(T.ceildiv(128, 64), T.ceildiv(64, 32), threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 64), T.float4_e2m1_unpacked)
            T.copy(A[by * 32, bx * 64], A_shared)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
