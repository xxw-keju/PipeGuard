# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_access_ptr_codegen.py#L38-L53
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.float16)
            for i in T.vectorized(4):
                T.ptx_cp_async(
                    T.access_ptr(S[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    1,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[0] = S[0]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
