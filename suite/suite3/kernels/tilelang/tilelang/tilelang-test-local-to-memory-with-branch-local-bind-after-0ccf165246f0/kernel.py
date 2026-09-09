# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_decouple_type_cast.py#L228-L242
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                if i < 8:
                    b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                if i_copy < 8:
                    b[i_copy] = b_local_cast[i_copy]
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
