# Generated Recovery Suite 2 real TileLang kernel. Do not edit.
# case_id=s2r-memory-tilelang-data-hazard-01
# independent_label=DATA_HAZARD; actors=thread:0,thread:64; variant=1
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        if tx == 0:
            Data[16] = T.int32(102)
        if tx == 64:
            Data[16] = T.int32(103)
            Out[1] = Data[16]
