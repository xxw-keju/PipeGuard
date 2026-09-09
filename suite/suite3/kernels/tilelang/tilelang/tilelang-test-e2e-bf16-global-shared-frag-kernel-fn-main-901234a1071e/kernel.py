# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_decouple_type_cast.py#L423-L433
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor((1024,), dtype=T.bfloat16),
        B: T.Tensor((1024,), dtype=T.bfloat16),
    ):
        with T.Kernel(1, threads=64):
            a_shared = T.alloc_shared((1024,), dtype=T.bfloat16)
            a_frag = T.alloc_fragment((1024,), dtype=T.float32)
            T.copy(A, a_shared)
            T.copy(a_shared, a_frag)
            T.copy(a_frag, B)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
