# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-sync-hazard-00
# independent_label=SYNC_HAZARD; actors=thread:0,thread:64; variant=0
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        if tx < 16:
            T.sync_threads()
        if tx == 0:
            Out[0] = T.int32(101)
