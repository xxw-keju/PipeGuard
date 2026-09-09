# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_decouple_type_cast.py#L114-L119
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # cond_buf is int32, acc is float32, but cond_buf is only in condition
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * 2.0, acc[i])
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
