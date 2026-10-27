# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-safe-02
# independent_label=SAFE; scenario=redundant-double-barrier; variant=2
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.int32)
        S[tx] = Data[tx]
        T.sync_threads()
        T.sync_threads()
        Out[tx] = S[127 - tx]
