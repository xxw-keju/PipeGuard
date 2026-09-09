# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_legalize_negative_index.py#L224-L239
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after(A: T.Tensor((T.int64(3000000000),), T.float32)):
        vec = T.Ramp(T.int32(-2), T.int32(1), 4)  # noqa: F841
        value = A[
            T.Shuffle(
                [
                    T.int64(2999999998),
                    T.int64(2999999999),
                    T.int64(0),
                    T.int64(1),
                ],
                [0, 1, 2, 3],
            )
        ]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
