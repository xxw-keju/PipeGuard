# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_func_attrs.py#L233-L243
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A, B, C):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B: T.Tensor[[m * 2 + 1], T.float32]
        C: T.Tensor[[m + 1], T.float32]
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
