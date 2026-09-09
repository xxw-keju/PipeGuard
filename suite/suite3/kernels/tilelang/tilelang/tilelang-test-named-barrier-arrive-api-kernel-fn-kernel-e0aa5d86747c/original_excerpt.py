        @T.prim_func
        def kernel(inp: T.Tensor((1,), T.int32), out: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=64) as _:
                tx = T.get_thread_binding()
                slot = T.alloc_shared((1,), T.int32)

                ready_barrier = 10
                empty_barrier = 11
                participant_threads = 64

                if tx < 32:
                    if tx == 0:
                        slot[0] = inp[0] + 1
                    T.named_barrier_arrive(ready_barrier, participant_threads)
                    T.sync_threads(empty_barrier, participant_threads)
                else:
                    T.sync_threads(ready_barrier, participant_threads)
                    if tx == 32:
                        out[0] = slot[0]
                    T.named_barrier_arrive(empty_barrier, participant_threads)
