# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_let_inline.py#L24-L29
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def expected(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    B[i, j] = A[i, j] * T.float32(2.0)
    return expected


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
