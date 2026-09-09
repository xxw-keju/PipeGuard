    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=256) as (bx,):
            tx = T.launch_thread("threadIdx.x", 256)
            barrier_id = tx % 4
            T.sync_threads(barrier_id, 256)
