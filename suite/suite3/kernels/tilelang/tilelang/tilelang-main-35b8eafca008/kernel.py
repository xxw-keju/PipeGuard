# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_2365.py#L8-L19
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

M, N, K = 128, 128, 33  # K = BLOCK_K + 1 -> 1-element residual K-tile

BM, BN, BK = 64, 64, 32

@T.prim_func
def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16), C: T.Tensor((M, N), T.float32)):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], As)
            T.copy(B[k * BK, bx * BN], Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, C[by * BM, bx * BN])


def _resolve_kernel(**pipeguard_bindings):
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
