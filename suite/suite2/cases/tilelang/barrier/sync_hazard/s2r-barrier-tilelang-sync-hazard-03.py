# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-sync-hazard-03
# independent_label=SYNC_HAZARD; scenario=elected-arrive-wrong-expected-count; variant=3
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        bar = T.alloc_barrier(128)
        if tx == 0:
            T.barrier_arrive(bar)
        T.mbarrier_wait_parity(bar, 0)
        Out[3] = T.int32(104)
