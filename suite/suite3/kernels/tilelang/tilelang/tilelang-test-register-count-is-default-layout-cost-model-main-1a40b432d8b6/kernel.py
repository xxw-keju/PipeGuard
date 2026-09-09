# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_layout_inference.py#L114-L124
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        S: T.Tensor((2,), T.float32),
        Out: T.Tensor((2, 2560), T.float32),
    ):
        with T.Kernel(1, threads=256):
            s_frag = T.alloc_fragment((2,), T.float32)
            for i in T.Parallel(2):
                s_frag[i] = S[i]
            for i, j in T.Parallel(2, 2560):
                Out[i, j] = s_frag[i] * 2.0
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
