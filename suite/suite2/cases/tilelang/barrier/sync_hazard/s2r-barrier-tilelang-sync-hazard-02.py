# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-sync-hazard-02
# independent_label=SYNC_HAZARD; scenario=circular-wait-two-mbarriers; variant=2
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        bar1 = T.alloc_barrier(64)
        bar2 = T.alloc_barrier(64)
        if tx < 64:
            T.mbarrier_wait_parity(bar1, 0)
            T.barrier_arrive(bar2)
        else:
            T.mbarrier_wait_parity(bar2, 0)
            T.barrier_arrive(bar1)
        Out[2] = T.int32(103)
