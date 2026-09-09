# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_cluster.py#L49-L57
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Tensor((128), T.int32)):
        with T.ClusterKernel(128, threads=128, cluster_dims=(4, 1, 1)):
            mbar = T.alloc_cluster_barrier([256])
            T.cluster_sync()
            T.mbarrier_arrive(mbar, 0)
            if T.block_rank_in_cluster() == 0:
                T.mbarrier_wait_parity(mbar, 0)
            T.cluster_sync()
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
