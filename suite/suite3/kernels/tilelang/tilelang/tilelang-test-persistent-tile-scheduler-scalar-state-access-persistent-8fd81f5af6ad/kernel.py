# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_tile_schedule.py#L6-L13
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def persistent(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, name="sched")
            sched.init(block_id)
            while sched.valid():
                A[sched.m_idx] = sched.n_idx + sched.current_iter
                sched.next_tile()
    return persistent


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
