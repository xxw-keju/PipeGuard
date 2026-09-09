# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/quantize/test_tilelang_quantize_fp8.py#L27-L35
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

from examples.dequantize_gemm.quantize.quantization import (
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_u8_to_f8_e4m3_to_f16_naive,
)

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def decode(
        values: T.Tensor((256,), "uint8"),
        naive: T.Tensor((256,), "uint16"),
        optimized: T.Tensor((256,), "uint16"),
    ):
        for i in T.serial(256):
            naive[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16_naive(8, values[i], T.float16), T.uint16)
            optimized[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16(8, values[i], T.float16), T.uint16)
    return decode


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
