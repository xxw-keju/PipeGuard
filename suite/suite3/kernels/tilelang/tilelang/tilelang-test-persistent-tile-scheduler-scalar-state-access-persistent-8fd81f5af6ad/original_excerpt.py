    @T.prim_func
    def persistent(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, name="sched")
            sched.init(block_id)
            while sched.valid():
                A[sched.m_idx] = sched.n_idx + sched.current_iter
                sched.next_tile()
