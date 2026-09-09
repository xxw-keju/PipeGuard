# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_1678.py#L10-L18
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def qwq_kernel():
        with T.Kernel(4096, 1, threads=1) as (pid_y, pid_x):
            i = T.alloc_var("int32")
            i = 1
            tmp_row = T.alloc_local((4,), "float32")
            amax_local = T.alloc_var("float32")
            j = 0
            amax_local = T.max(amax_local, tmp_row[j])
    return qwq_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
