# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/target/test_tilelang_codegen_cutedsl_integer_shift.py#L40-L45
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def prog(A: T.Tensor((1,), "int8"), B: T.Tensor((1,), "uint16")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "int8")
            local[0] = A[0]
            B[0] = T.cast(local[0] >> 1, T.uint16)
    return prog


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
