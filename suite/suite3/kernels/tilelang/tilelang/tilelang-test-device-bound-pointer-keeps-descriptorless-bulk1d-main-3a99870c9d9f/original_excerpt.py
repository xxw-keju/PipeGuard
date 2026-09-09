    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out: T.Tensor((256,), T.float16),
    ):
        with T.Kernel(threads=128):
            src = T.make_tensor(src_ptrs[0], (256,), T.float16)
            src_shared = T.alloc_shared((256,), T.float16)
            T.copy(src, src_shared, prefer_instruction="tma")
            T.copy(src_shared, out, prefer_instruction="sync")
