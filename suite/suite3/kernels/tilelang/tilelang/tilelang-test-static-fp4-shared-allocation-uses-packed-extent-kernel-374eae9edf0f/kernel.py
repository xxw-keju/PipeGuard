# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/amd/test_tilelang_mxfp4_gfx950.py#L59-L63
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn, scope="shared")
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
