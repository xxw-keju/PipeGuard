    @T.prim_func
    def decode(
        values: T.Tensor((256,), "uint8"),
        naive: T.Tensor((256,), "uint16"),
        optimized: T.Tensor((256,), "uint16"),
    ):
        for i in T.serial(256):
            naive[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16_naive(8, values[i], T.float16), T.uint16)
            optimized[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16(8, values[i], T.float16), T.uint16)
