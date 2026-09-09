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
