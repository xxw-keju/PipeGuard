    @T.prim_func(private=True)
    def func():
        A_shared = T.alloc_buffer((256,), dtype="float32", scope="shared")
        idx = T.alloc_buffer((16,), dtype="int32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        # Bounded pointer access covering the whole buffer (fill-style).
        T.evaluate(
            T.call_extern(
                "handle",
                "fake_fill",
                T.tvm_access_ptr(T.type_annotation("float32"), A_shared.data, 0, 256, 2),
            )
        )
        # Data-dependent atomic destination: the touched interval relaxes to
        # everything once the enclosing loop is summarized.
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "float32",
                    tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                    T.address_of(A_shared[idx[i]]),
                    T.float32(1),
                    T.int32(0),
                )
            )
