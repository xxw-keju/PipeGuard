# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-safe-03
# independent_label=SAFE; scenario=named-barrier-exact-scope; variant=3
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((64,), T.int32)
        if tx < 64:
            S[tx] = Data[tx]
            T.sync_threads(1, 64)
            Out[tx] = S[63 - tx]
