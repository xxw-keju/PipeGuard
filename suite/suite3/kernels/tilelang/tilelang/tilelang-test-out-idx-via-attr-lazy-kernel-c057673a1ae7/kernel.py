# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_func_attrs.py#L14-L22
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A):
        A: T.Tensor[[128, 128], T.float32]
        B = T.empty([128, 128], T.float32)
        with T.Kernel(1):
            for i in T.serial(128):
                for j in T.serial(128):
                    B[i, j] = A[i, j] + 1.0
        return B
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
