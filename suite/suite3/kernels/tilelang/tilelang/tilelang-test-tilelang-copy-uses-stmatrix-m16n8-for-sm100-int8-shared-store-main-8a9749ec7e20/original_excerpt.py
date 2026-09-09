    @T.prim_func
    def main(A: T.Tensor((16, 8), T.int8)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((16, 8), T.int8)
            smem = T.alloc_shared((16, 8), T.int8)

            T.annotate_layout({frag: make_gemm_fragment_8x8_transposed().repeat([2, 1], repeat_on_thread=False)})

            for i, j in T.Parallel(16, 8):
                frag[i, j] = A[i, j]

            T.copy(frag, smem)
