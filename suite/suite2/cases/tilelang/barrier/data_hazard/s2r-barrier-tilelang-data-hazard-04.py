# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-barrier-tilelang-data-hazard-04
# independent_label=DATA_HAZARD; scenario=barrier-after-the-handoff; variant=4
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        shared = T.alloc_shared((8,), T.int32)
        if tx == 0:
            shared[0] = T.int32(101)
        if tx == 64:
            Out[5] = shared[0]
        T.sync_threads()
