# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_materialize_ws_schedule.py#L1924-L1938
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=4,
                    roles=[
                        T.WSRole("R0", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("R1", warps_lo=1, warps_hi=2, max_nreg=224),
                    ],
                    pipelines=[],
                    scopes=[T.WSScope(T.WSScope.ROOT, {"R0": ["cp"], "R1": []})],
                )
            )
            T.copy(A, B, annotations={T.WSID: "cp"})
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
