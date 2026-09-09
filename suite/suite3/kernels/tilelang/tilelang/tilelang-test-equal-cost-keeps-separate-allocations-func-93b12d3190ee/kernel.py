# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_tmem_arena_packing.py#L131-L139
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            S_tmem = T.alloc_tmem([128, 128], T.float32)
            P_tmem = T.alloc_tmem([128, 128], T.float16)
            O_tmem = T.alloc_tmem([128, 128], T.float32)
            T.evaluate(S_tmem[0, 0])
            T.evaluate(P_tmem[0, 0])
            T.evaluate(O_tmem[0, 0])
    return func


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
