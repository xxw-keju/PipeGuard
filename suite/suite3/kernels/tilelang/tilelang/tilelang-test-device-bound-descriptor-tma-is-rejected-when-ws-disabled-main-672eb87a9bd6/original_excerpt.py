    @T.prim_func
    def main(src_ptrs: T.Tensor((1,), T.ptr)):
        with T.Kernel(threads=128):
            src = T.make_tensor(
                src_ptrs[0],
                (16, 32),
                T.float16,
                strides=(32, 1),
            )
            src_shared = T.alloc_shared((16, 16), T.float16)
            T.copy(src[0, 0], src_shared, prefer_instruction="tma")
