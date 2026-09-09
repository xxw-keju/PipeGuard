# Generated Suite 2 Barrier case; do not hand edit.
# case_id=s2r-barrier-tilelang-safe-03
# scenario=BARRIER:SHARED-DELAYED-RAW; independent_label=SAFE
import tilelang.language as T

@T.prim_func
def kernel(Data: T.Tensor((256,), T.int32), Out: T.Tensor((128,), T.int32)):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        shared = T.alloc_shared((2,), T.int32)
        if tx == 0:
            shared[0] = T.int32(1)
            for step in T.serial(64):
                shared[1] = T.int32(step)
        T.sync_threads()
        if tx == 64:
            Out[4] = shared[0]
