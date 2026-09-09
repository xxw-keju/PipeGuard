# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py#L558-L572
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def expected(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[
                    T.Shuffle(
                        [
                            T.Select(t < 2, t + 1022, t - 2),
                            T.Select(t < 1, t + 1023, t - 1),
                            t,
                            t + 1,
                        ],
                        [0, 1, 2, 3],
                    )
                ]
    return expected


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
