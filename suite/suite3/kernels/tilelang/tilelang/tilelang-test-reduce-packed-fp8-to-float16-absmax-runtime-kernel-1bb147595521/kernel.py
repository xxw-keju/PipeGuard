# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_reduce.py#L1113-L1120
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((4, 32), T.float8_e4m3fn), B: T.Tensor((4,), T.float16)):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((4, 32), T.float8_e4m3fn)
            dst = T.alloc_fragment((4,), T.float16)
            T.copy(A, src)
            T.reduce_absmax(src, dst, dim=1)
            T.copy(dst, B)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
