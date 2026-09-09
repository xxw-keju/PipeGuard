# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/target/test_tilelang_codegen_cutedsl_cp_async.py#L51-L56
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]
    return prog


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
