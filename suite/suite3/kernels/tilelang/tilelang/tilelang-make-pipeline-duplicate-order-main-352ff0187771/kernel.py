# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_span.py#L203-L214
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), "float16")
            B_shared = T.alloc_shared((128, 128), "float16")
            for _k in T.Pipelined(2, order=[0, 0], stage=[0, 1]):  # span_marker_pipelined
                T.copy(A, A_shared)  # span_marker_pipe_copy1
                T.copy(B, B_shared)  # span_marker_pipe_copy2
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
