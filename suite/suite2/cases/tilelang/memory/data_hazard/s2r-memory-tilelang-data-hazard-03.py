# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-data-hazard-03
# independent_label=DATA_HAZARD; scenario=mbarrier-arrive-before-publish; variant=3
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.int32)
        bar = T.alloc_barrier(128)
        T.barrier_arrive(bar)
        S[tx] = Data[tx]
        T.mbarrier_wait_parity(bar, 0)
        Out[tx] = S[127 - tx]
