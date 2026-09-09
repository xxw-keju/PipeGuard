# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-sync-hazard-04
# independent_label=SYNC_HAZARD; actors=thread:0,thread:64; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        if tx == 0:
            Data[40] = T.int32(105)
        if tx < 96:
            T.sync_threads()
        if tx == 64:
            Out[4] = Data[40]
