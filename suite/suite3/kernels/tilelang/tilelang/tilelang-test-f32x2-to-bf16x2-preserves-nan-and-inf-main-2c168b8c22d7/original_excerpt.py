    @T.prim_func
    def main(bits: T.Tensor((4,), "uint32"), out: T.Tensor((4,), "float32")):
        with T.Kernel(2, threads=1) as bx:
            f0 = T.reinterpret(bits[bx * 2], T.float32)
            f1 = T.reinterpret(bits[bx * 2 + 1], T.float32)
            packed = _tir_f32x2_to_bf16x2_to_u32(f0, f1)
            decoded = list(_tir_u32_to_bf16x2_to_f32x2(packed))
            out[bx * 2] = decoded[0]
            out[bx * 2 + 1] = decoded[1]
