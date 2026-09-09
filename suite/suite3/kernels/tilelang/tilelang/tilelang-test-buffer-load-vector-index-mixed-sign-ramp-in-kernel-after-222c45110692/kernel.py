# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_legalize_negative_index.py#L256-L260
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32), B: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            vec = T.Ramp(-2, 1, 4)  # noqa: F841
            B[T.Ramp(0, 1, 4)] = A[T.Shuffle([1022, 1023, 0, 1], [0, 1, 2, 3])]
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
