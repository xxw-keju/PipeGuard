    @T.prim_func
    def expected(
        flag: T.Buffer((2,), "int32"),
        counter_tvm: T.Buffer((1,), "int32"),
        counter_tl: T.Buffer((1,), "int32"),
        out: T.Buffer((4,), "int32"),
    ):
        if flag[0] != 0:
            snapshot_tvm = T.bind(counter_tvm[0])
            pos_tvm = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    counter_tvm.access_ptr("rw"),
                    1,
                )
            )
            out[0] = snapshot_tvm
            out[1] = pos_tvm
        if flag[1] != 0:
            snapshot_tl = T.bind(counter_tl[0])
            pos_tl = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    T.call_intrin(
                        "handle",
                        tvm.ir.Op.get("tl.access_ptr"),
                        counter_tl[0],
                        1,
                        3,
                    ),
                    1,
                )
            )
            out[2] = snapshot_tl
            out[3] = pos_tl
