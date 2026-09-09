# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_1026.py#L8-L15
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def shared_kernel():
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared_mem = T.alloc_shared((32), dtype="float32", scope="shared")
            if tx % 2 == 0:
                a = shared_mem[tx]
                shared_mem[tx ^ 1] = a
    return shared_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
