# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_lower_hopper_intrin.py#L95-L126
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before():
        T.func_attr({"tirx.is_entry_func": True, "tl.has_tma": T.bool(True)})
        Output_partial = T.alloc_buffer((32,), "float16")
        with T.launch_thread("threadIdx.x", 1):
            T.evaluate(
                T.create_tma_descriptor(
                    6,
                    4,
                    Output_partial.data,
                    8,
                    2,
                    2,
                    1,
                    2,
                    16,
                    32,
                    64,
                    8,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
