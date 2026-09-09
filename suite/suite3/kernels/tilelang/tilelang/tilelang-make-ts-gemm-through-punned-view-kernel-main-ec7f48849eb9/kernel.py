# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_tcgen05_gemm.py#L159-L182
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        P: T.Tensor((128, 128), T.bfloat16),
        V: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            P_shared = T.alloc_shared((128, 128), T.bfloat16)
            V_shared = T.alloc_shared((128, 128), T.bfloat16)
            P_frag = T.alloc_fragment((128, 128), T.bfloat16)
            O_frag = T.alloc_fragment((128, 128), T.float32)
            S_tmem = T.alloc_tmem((128, 128), T.float32)
            O_tmem = T.alloc_tmem((128, 128), T.float32)
            P_view = T.view(S_tmem, shape=(128, 256), dtype=T.bfloat16)
            done = T.alloc_barrier(1)

            T.copy(P, P_shared)
            T.copy(P_shared, P_frag)
            T.copy(P_frag, P_view[:, 128:256])
            T.copy(V, V_shared)
            T.gemm(P_view[:, 128:256], V_shared, O_tmem, mbar=done, clear_accum=True)
            T.mbarrier_wait_parity(done, 0)
            T.copy(O_tmem, O_frag)
            T.copy(O_frag, D)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
