# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_span.py#L229-L235
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((4,), "float32")  # span_marker_frag_alloc2
            for i in T.Parallel(4):
                frag[0] = A[i]
                frag[1] = A[i]  # span_marker_frag_store2
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
