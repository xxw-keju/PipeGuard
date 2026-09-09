# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_inject_set_max_nreg.py#L135-L155
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before_no_set_max_nreg(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        v = T.launch_thread("threadIdx.x", 128)

        with T.sblock(""):
            T.reads(A[bx * 64, 0:64])
            T.writes()

            # Add no_set_max_nreg to disable register hinting
            T.disable_warp_group_reg_alloc()

            mbars = T.alloc_barrier([128, 128])  # noqa: F841
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                # Producer branch - should not have set_max_nreg calls
                T.evaluate(0)
            else:
                # Consumer branch - should not have set_max_nreg calls
                T.evaluate(0)
    return before_no_set_max_nreg


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
