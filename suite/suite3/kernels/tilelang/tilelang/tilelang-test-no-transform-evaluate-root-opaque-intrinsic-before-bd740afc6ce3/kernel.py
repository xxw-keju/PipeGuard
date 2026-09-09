# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_decouple_type_cast.py#L585-L593
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
