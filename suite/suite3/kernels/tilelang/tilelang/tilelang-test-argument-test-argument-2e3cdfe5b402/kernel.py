# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_frontend_v2.py#L16-L43
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def test_argument(
        t_1: T.bool,
        t_2: T.short,
        t_3: T.int,
        t_4: T.long,
        t_5: T.half,
        t_6: T.float,
        t_7: T.long,
        t_8: T.int8,
        t_9: T.int16,
        t_10: T.int32,
        t_11: T.int64,
        t_12: T.uint8,
        t_13: T.uint16,
        t_14: T.uint32,
        t_15: T.uint64,
        t_16: T.float8_e4m3fn,
        t_17: T.float8_e4m3fnuz,
        t_18: T.float8_e5m2,
        t_19: T.float8_e5m2fnuz,
        t_20: T.float8_e8m0fnu,
        t_21: T.float16,
        t_22: T.bfloat16,
        t_23: T.float32,
        t_24: T.float64,
    ):
        pass
    return test_argument


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
