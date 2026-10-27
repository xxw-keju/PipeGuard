# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-data-hazard-02
# independent_label=DATA_HAZARD; scenario=named-barrier-wrong-scope; variant=2
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((64,), T.int32)
        if tx < 64:
            S[tx] = Data[tx]
            T.sync_threads(1, 64)
        else:
            Out[tx - 64] = S[tx - 64]
