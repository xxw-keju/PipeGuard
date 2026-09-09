    @T.prim_func
    def stateless(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, stateful=False)
            bx, by = sched.coord(block_id)
            A[bx] = by
