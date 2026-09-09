# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_tmem_copy.py#L418-L428
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            frag16 = T.alloc_fragment((128, 64), T.bfloat16)
            frag32 = T.alloc_fragment((128, 32), T.float32)
            tmem = T.alloc_tmem((128, 64), T.bfloat16)
            view = T.view(tmem, shape=(128, 32), dtype=T.float32)

            T.annotate_layout({tmem: T.Layout((128, 64), lambda i, j: [i, 2 * j])})
            T.copy(frag16, tmem)
            T.copy(view, frag32)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
