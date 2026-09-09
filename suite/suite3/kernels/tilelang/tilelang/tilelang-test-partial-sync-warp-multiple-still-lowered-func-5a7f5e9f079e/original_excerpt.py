    @T.prim_func(private=True)
    def func():
        S = T.alloc_buffer((64,), dtype="float32", scope="shared")
        acc = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        # 32 participating threads: exactly one warp.
        if tx < 32:
            acc[0] = T.float32(0)
            for i in range(2):
                S[tx] = T.float32(1)
                acc[0] += S[31 - tx]
