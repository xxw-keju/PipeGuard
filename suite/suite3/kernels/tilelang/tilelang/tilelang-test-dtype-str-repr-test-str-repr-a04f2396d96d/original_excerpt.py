    @T.prim_func
    def test_str_repr():
        buf_1 = T.alloc_buffer((1,), dtype=T.bool, scope="shared")  # noqa F841
        buf_2 = T.alloc_buffer((1,), dtype=T.short, scope="shared")  # noqa F841
        buf_3 = T.alloc_buffer((1,), dtype=T.int, scope="shared")  # noqa F841
        buf_4 = T.alloc_buffer((1,), dtype=T.long, scope="shared")  # noqa F841
        buf_5 = T.alloc_buffer((1,), dtype=T.half, scope="shared")  # noqa F841
        buf_6 = T.alloc_buffer((1,), dtype=T.float, scope="shared")  # noqa F841
        buf_7 = T.alloc_buffer((1,), dtype=T.long, scope="shared")  # noqa F841
        buf_8 = T.alloc_buffer((1,), dtype=T.int8, scope="shared")  # noqa F841
        buf_9 = T.alloc_buffer((1,), dtype=T.int16, scope="shared")  # noqa F841
        buf_10 = T.alloc_buffer((1,), dtype=T.int32, scope="shared")  # noqa F841
        buf_11 = T.alloc_buffer((1,), dtype=T.int64, scope="shared")  # noqa F841
        buf_12 = T.alloc_buffer((1,), dtype=T.uint8, scope="shared")  # noqa F841
        buf_13 = T.alloc_buffer((1,), dtype=T.uint16, scope="shared")  # noqa F841
        buf_14 = T.alloc_buffer((1,), dtype=T.uint32, scope="shared")  # noqa F841
        buf_15 = T.alloc_buffer((1,), dtype=T.uint64, scope="shared")  # noqa F841
        buf_16 = T.alloc_buffer((1,), dtype=T.float8_e4m3fn, scope="shared")  # noqa F841
        buf_17 = T.alloc_buffer((1,), dtype=T.float8_e4m3fnuz, scope="shared")  # noqa F841
        buf_18 = T.alloc_buffer((1,), dtype=T.float8_e5m2, scope="shared")  # noqa F841
        buf_19 = T.alloc_buffer((1,), dtype=T.float8_e5m2fnuz, scope="shared")  # noqa F841
        buf_20 = T.alloc_buffer((1,), dtype=T.float8_e8m0fnu, scope="shared")  # noqa F841
        buf_21 = T.alloc_buffer((1,), dtype=T.float16, scope="shared")  # noqa F841
        buf_22 = T.alloc_buffer((1,), dtype=T.bfloat16, scope="shared")  # noqa F841
        buf_23 = T.alloc_buffer((1,), dtype=T.float32, scope="shared")  # noqa F841
        buf_24 = T.alloc_buffer((1,), dtype=T.float64, scope="shared")  # noqa F841
