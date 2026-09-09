# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_inject_set_max_nreg.py#L102-L119
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A: T.Tensor((512, 512), T.float16), B: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        v = T.launch_thread("threadIdx.x", 256)

        with T.sblock(""):
            T.reads(A[bx * 64, 0:64])
            T.writes(B[bx * 64, 0:64])

            A_shared = T.alloc_buffer((128,), T.float16, scope="shared")
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                T.set_max_nreg(80, 0)
                A_shared[v - 128] = A[bx * 64, v - 128]
            else:
                T.set_max_nreg(240, 1)
                B[bx * 64, v] = A_shared[v]
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
