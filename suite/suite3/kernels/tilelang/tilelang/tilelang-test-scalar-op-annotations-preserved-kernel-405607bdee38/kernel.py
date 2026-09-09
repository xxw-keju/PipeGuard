# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_materialize_ws_schedule.py#L135-L161
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[T.WSRole("Worker", warps_lo=0, warps_hi=4, max_nreg=224)],
                    pipelines=[],
                    scopes=[
                        T.WSScope(
                            T.WSScope.ROOT,
                            {
                                "Worker": [
                                    "copy",
                                    "atomic_max",
                                    "atomic_min",
                                    "atomic_add",
                                ]
                            },
                        )
                    ],
                )
            )
            T.copy(A[0], B[0], annotations={T.WSID: "copy"})
            T.atomic_max(B[0], A[0], annotations={T.WSID: "atomic_max"})
            T.atomic_min(B[0], A[0], annotations={T.WSID: "atomic_min"})
            T.atomic_add(B[0], A[0], annotations={T.WSID: "atomic_add"})
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
