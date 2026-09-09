# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/cpu/test_tilelang_cpu_while_repro.py#L21-L30
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(flag: T.Buffer((1,), "int32"), out: T.Buffer((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
