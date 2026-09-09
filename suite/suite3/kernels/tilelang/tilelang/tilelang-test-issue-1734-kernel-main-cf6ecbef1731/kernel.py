# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_1734.py#L11-L35
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor[(2, 512), T.float32],
        B: T.Tensor[(2, 512), T.float32],
        C: T.Tensor[(2,), T.float32],
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((2, 512), T.float32)
            B_local = T.alloc_fragment((2, 512), T.float32)
            C_local = T.alloc_fragment((2,), T.float32)

            # Each thread owns 8 contiguous elements of one row, so i is
            # constant per thread and the guard below is loop-invariant,
            # while a residual serial loop survives vectorization.
            row_chunks = T.Fragment((2, 512), forward_fn=lambda i, j: (i * 64 + j // 8, j % 8))
            T.annotate_layout({A_local: row_chunks, B_local: row_chunks})

            T.copy(A, A_local)
            T.copy(C, C_local)

            for i, j in T.Parallel(2, 512):
                if C_local[i] >= 0:
                    B_local[i, j] = A_local[i, j]

            T.copy(B_local, B)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
