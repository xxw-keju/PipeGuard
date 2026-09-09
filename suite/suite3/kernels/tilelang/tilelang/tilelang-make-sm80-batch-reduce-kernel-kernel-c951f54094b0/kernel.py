# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_reduce.py#L321-L331
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(
        A: T.Tensor((128, 64), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=256):
            src = T.alloc_shared((128, 64), T.float32)
            dst = T.alloc_fragment((128,), T.float32)
            T.copy(A, src, disable_tma=True)
            T.reduce_sum(src, dst, dim=1, batch=4)
            T.copy(dst, B)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
