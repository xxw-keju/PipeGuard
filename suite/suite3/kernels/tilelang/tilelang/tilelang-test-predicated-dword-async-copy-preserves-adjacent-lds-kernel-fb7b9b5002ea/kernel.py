# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/amd/test_tilelang_hip_codegen.py#L841-L859
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(
        A: T.Tensor((1,), T.float32),
        B: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4,), T.float32)
            for i in T.serial(4):
                A_shared[i] = T.float32(i + 1)
            T.ptx_cp_async(
                T.access_ptr(A_shared[0], "w", 1),
                T.access_ptr(A[0], "r", 1),
                1,
                predicate=False,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            for i in T.serial(4):
                B[i] = A_shared[i]
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
