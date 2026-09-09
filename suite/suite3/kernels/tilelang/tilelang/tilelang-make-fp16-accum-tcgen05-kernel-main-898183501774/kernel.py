# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_tcgen05_gemm.py#L667-L687
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float8_e4m3fn),
        B: T.Tensor((128, 128), T.float8_e4m3fn),
        D: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            B_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            C_tmem = T.alloc_tmem((128, 128), T.float16)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float16)
            C_shared = T.alloc_shared((128, 128), T.float16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, clear_accum=True)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
