# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-safe-04
# independent_label=SAFE; scenario=two-phase-mbarrier-parity-flip; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.int32)
        R = T.alloc_shared((128,), T.int32)
        bar = T.alloc_barrier(128)
        S[tx] = Data[tx]
        T.barrier_arrive(bar)
        T.mbarrier_wait_parity(bar, 0)
        R[tx] = S[127 - tx]
        T.barrier_arrive(bar)
        T.mbarrier_wait_parity(bar, 1)
        Out[tx] = R[127 - tx]
