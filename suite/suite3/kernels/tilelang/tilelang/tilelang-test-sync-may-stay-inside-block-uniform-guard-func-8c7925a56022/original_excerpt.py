    @T.prim_func(private=True)
    def func(flag: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if flag < 0:
            s[tx] = T.cast(tx, "float32")
            x1: T.float32 = s[(tx + 64) % 128]
            s[tx] = x1 * T.float32(2)
