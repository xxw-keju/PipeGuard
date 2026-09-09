# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/target/test_tilelang_codegen_cutedsl_reduce.py#L21-L36
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def prog(A: T.Tensor((1, 512), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            x_frag = T.alloc_fragment((1, 512), "float32")
            sum_frag = T.alloc_fragment((1,), "float32")
            T.annotate_layout(
                {
                    x_frag: T.Fragment(x_frag.shape, forward_fn=lambda i, j: (j // 8, j % 8)),
                    sum_frag: T.Fragment(sum_frag.shape, forward_fn=lambda i, rep: (rep, 0), replicate=64),
                }
            )
            for i, j in T.Parallel(1, 512):
                x_frag[i, j] = A[i, j]
            T.reduce_sum(x_frag, sum_frag, dim=1)
            for i in T.Parallel(1):
                B[i] = sum_frag[i]
    return prog


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
