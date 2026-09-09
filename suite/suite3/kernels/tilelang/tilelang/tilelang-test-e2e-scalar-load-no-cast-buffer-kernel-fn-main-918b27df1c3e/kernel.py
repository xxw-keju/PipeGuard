# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_decouple_type_cast.py#L551-L562
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        a: T.Tensor[(32,), T.float8_e4m3fn],
        b: T.Tensor[(1,), T.float32],
        c: T.Tensor[(32,), T.float8_e4m3fn],
    ):
        with T.Kernel(1, threads=32):
            a_local = T.alloc_local((32,), T.float8_e4m3fn)
            T.copy(a, a_local)

            for i in T.vectorized(32):
                c[i] = a_local[i] * b[0]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
