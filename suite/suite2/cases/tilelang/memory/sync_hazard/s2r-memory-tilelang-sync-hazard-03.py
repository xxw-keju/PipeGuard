# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-sync-hazard-03
# independent_label=SYNC_HAZARD; scenario=loop-constant-parity-overshoot; variant=3
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.int32)
        bar = T.alloc_barrier(128)
        S[tx] = Data[tx]
        for i in T.serial(4):
            T.barrier_arrive(bar)
            T.mbarrier_wait_parity(bar, 0)
            S[tx] = S[127 - tx] + 1
        Out[tx] = S[tx]
