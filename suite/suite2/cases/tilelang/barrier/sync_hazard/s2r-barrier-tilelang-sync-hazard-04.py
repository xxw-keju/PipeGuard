# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-sync-hazard-04
# independent_label=SYNC_HAZARD; scenario=wait-before-any-arrival; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        bar = T.alloc_barrier(128)
        T.mbarrier_wait_parity(bar, 0)
        T.barrier_arrive(bar)
        Out[4] = T.int32(105)
