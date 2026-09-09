    @T.prim_func(private=True)
    def func(idx_buf: T.Buffer((128,), "int32")):
        s = T.alloc_buffer((256,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        a: T.int32 = idx_buf[tx]
        s[a] = T.cast(tx, "float32")
        l[0] = s[a + 1]
