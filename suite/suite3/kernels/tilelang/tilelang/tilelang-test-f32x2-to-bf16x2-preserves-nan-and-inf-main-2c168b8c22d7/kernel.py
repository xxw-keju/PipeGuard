# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_bf16x2_pack_special_values.py#L16-L24
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

from examples.dequantize_gemm.quantize.quantization import (
    _tir_f32x2_to_bf16x2_to_u32,
    _tir_u32_to_bf16x2_to_f32x2,
)

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(bits: T.Tensor((4,), "uint32"), out: T.Tensor((4,), "float32")):
        with T.Kernel(2, threads=1) as bx:
            f0 = T.reinterpret(bits[bx * 2], T.float32)
            f1 = T.reinterpret(bits[bx * 2 + 1], T.float32)
            packed = _tir_f32x2_to_bf16x2_to_u32(f0, f1)
            decoded = list(_tir_u32_to_bf16x2_to_f32x2(packed))
            out[bx * 2] = decoded[0]
            out[bx * 2 + 1] = decoded[1]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
