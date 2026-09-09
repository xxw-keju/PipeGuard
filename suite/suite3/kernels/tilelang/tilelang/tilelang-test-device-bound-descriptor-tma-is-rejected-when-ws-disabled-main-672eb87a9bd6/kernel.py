# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_tma_copy.py#L727-L737
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(src_ptrs: T.Tensor((1,), T.ptr)):
        with T.Kernel(threads=128):
            src = T.make_tensor(
                src_ptrs[0],
                (16, 32),
                T.float16,
                strides=(32, 1),
            )
            src_shared = T.alloc_shared((16, 16), T.float16)
            T.copy(src[0, 0], src_shared, prefer_instruction="tma")
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
