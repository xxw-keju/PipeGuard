# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-safe-04
# independent_label=SAFE; scenario=elected-mbarrier-publication; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((1,), T.int32)
        bar = T.alloc_barrier(1)
        if tx == 0:
            S[0] = Data[0]
            T.barrier_arrive(bar)
        T.mbarrier_wait_parity(bar, 0)
        Out[tx] = S[0]
