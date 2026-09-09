# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_scan.py#L53-L61
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def scan(A: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            sum_frag = T.alloc_fragment((32,), T.float32)
            max_frag = T.alloc_fragment((32,), T.float32)
            T.copy(A, sum_frag)
            T.copy(A, max_frag)
            T.cumsum(sum_frag, annotations={"test.scan": "cumsum"})
            T.cummax(max_frag, annotations={"test.scan": "cummax"})
    return scan


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
