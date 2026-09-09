# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_span.py#L35-L49
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:  # span_marker_kernel
            A_shared = T.alloc_shared((128, 128), "float16")  # span_marker_alloc_shared
            C_local = T.alloc_fragment((128, 128), "float32")  # span_marker_alloc_fragment
            T.copy(A, A_shared)  # span_marker_copy
            T.clear(C_local)  # span_marker_clear
            for _k in T.serial(8):  # span_marker_for
                T.gemm(A_shared, A_shared, C_local)  # span_marker_gemm
            if bx == 0:  # span_marker_if
                T.copy(C_local, C)  # span_marker_copy_out
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
