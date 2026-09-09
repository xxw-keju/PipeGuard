# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_view.py#L128-L142
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), T.float16),
        B: T.Tensor((64, 128), T.int8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_stage = T.alloc_shared((2, 64, 64), T.float16, scope="shared.dyn")
            A_i8 = T.view(A_stage, (2, 64, 128), dtype=T.int8)
            T.annotate_layout({A_i8: T.Layout((2, 64, 128), lambda s, i, j: [s, i, j])})

            for i, j in T.Parallel(64, 64):
                A_stage[0, i, j] = A[i, j]

            for i, j in T.Parallel(64, 128):
                B[i, j] = A_i8[0, i, j]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
