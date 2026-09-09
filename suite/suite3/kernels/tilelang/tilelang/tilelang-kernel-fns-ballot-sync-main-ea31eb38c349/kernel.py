# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_warp_vote.py#L223-L230
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.cast(T.ballot_sync(tx >= 7), "uint32")
            B[tx] = T.cast(T.__fns(mask, 0, 3), "int32")
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
