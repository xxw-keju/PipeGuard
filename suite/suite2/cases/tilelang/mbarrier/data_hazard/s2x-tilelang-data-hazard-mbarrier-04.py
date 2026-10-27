# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2x-tilelang-data-hazard-mbarrier-04
# independent_label=DATA_HAZARD; scenario=dead-guarded-protocol; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((32,), T.int32)):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        S = T.alloc_shared((32,), T.int32)
        bar = T.alloc_barrier(32)
        S[tx] = Data[tx]
        if tx < 0:
            T.barrier_arrive(bar)
            T.mbarrier_wait_parity(bar, 0)
        Out[tx] = S[31 - tx]
