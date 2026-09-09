# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/metal/test_metal_address_space.py#L21-L41
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

M = N = K = 128

BLOCK_M = BLOCK_N = 16

BLOCK_K = 8

THREADS = 64

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=THREADS,
        ) as (bx, by):
            As = T.alloc_shared((BLOCK_M, BLOCK_K), T.float16)
            Bs = T.alloc_shared((BLOCK_K, BLOCK_N), T.float16)
            Cl = T.alloc_shared((BLOCK_M, BLOCK_N), T.float32)

            T.clear(Cl)
            for k in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], As)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * BLOCK_M, bx * BLOCK_N])
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
