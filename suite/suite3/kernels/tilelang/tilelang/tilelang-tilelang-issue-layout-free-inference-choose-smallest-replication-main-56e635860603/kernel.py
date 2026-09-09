# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/layout/test_tilelang_layout_inference.py#L13-L27
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Tensor((128, 4), T.float), B: T.Tensor((128, 4), T.float)):
        with T.Kernel(1, threads=128) as _:
            A_frag = T.alloc_fragment((128, 4), T.float)
            B_frag = T.alloc_fragment((128, 4), T.float)
            S_frag = T.alloc_fragment((4,), T.float)
            T.annotate_layout(
                {
                    A_frag: T.Fragment(A_frag.shape, lambda i, j: (i, j)),
                }
            )
            for i, j in T.Parallel(128, 4):
                A_frag[i, j] = S_frag[j]
            for i, j in T.Parallel(128, 4):
                B_frag[i, j] = S_frag[j]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
