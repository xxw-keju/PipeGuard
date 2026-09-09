    @T.prim_func
    def func():
        buf_dyn_shmem = T.alloc_buffer((98304,), "uint8", scope="shared.dyn")
        tx = T.launch_thread("threadIdx.x", 384)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in T.unroll(8):
            off = (
                i // 4 * 8192
                + tx // 32 * 1024
                + tx % 16 * 64
                + (tx % 8 // 4 + i % 4 // 2) % 2 * 32
                + (tx % 4 // 2 + i % 2) % 2 * 16
                + (tx % 32 // 16 + tx % 2) % 2 * 8
            )
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.tirx.op.Op.get("tl.ptx_stmatrix"),
                    T.int32(0),
                    T.int32(4),
                    T.tvm_access_ptr(
                        T.type_annotation("uint8"),
                        buf_dyn_shmem.data,
                        off,
                        98304 - off,
                        2,
                    ),
                    T.int32(2),
                )
            )
