    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=256) as (bx,):
            barrier_id = T.int32(1)
            T.sync_threads(barrier_id)
            T.sync_threads(barrier_id, 128)
