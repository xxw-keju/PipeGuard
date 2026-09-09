    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), counter: T.Buffer((1,), "int32"), out: T.Buffer((4,), "int32")):
        if A[0] >= T.float32(0):
            pos = T.bind(
                T.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.atomic_add_ret_elem_op"),
                    counter.access_ptr("rw"),
                    1,
                )
            )
            if pos < 4:
                out[pos] = 1
