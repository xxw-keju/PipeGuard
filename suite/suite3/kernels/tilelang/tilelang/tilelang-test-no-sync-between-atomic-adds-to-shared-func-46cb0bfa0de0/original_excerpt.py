    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((16, 128), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.tvm_access_ptr(
                        T.type_annotation("float32"),
                        A_shared.data,
                        i * 128 + tx,
                        1,
                        3,
                    ),
                    T.float32(1),
                    T.int32(0),
                )
            )
