# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_access_ptr_codegen.py#L118-L129
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S0 = T.alloc_shared((4,), T.float16)
            S1 = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S0)
            # shared->shared cannot use cp.async and should fail for async_copy.
            T.async_copy(S0, S1)
            T.copy(S1, B[0:4])
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
