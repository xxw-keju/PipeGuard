# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2x-tilelang-data-hazard-mbarrier-02
# independent_label=DATA_HAZARD; scenario=loop-same-phase-neighbor-race; variant=2
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((32,), T.int32)):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        S = T.alloc_shared((32,), T.int32)
        bar = T.alloc_barrier(32)
        S[tx] = Data[tx]
        for i in T.serial(2):
            T.barrier_arrive(bar)
            T.mbarrier_wait_parity(bar, i % 2)
            S[tx] = S[31 - tx] + 1
        Out[tx] = S[tx]
