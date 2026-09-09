# Generated Suite 2 Barrier case; do not hand edit.
# case_id=s2r-barrier-tilelang-data-hazard-01
# scenario=BARRIER:SHARED-WAW; independent_label=DATA_HAZARD
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        shared = T.alloc_shared((8,), T.int32)
        if tx == 0:
            shared[0] = T.int32(101)
        if tx == 64:
            shared[0] = T.int32(202)
            Out[1] = shared[0]
