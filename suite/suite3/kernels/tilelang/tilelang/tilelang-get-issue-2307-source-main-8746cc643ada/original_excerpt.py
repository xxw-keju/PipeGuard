    @T.prim_func
    def main(score: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=32):
            score_fragment = T.alloc_fragment(32, T.float32)
            for i in T.Parallel(32):
                if i < 16:
                    score_fragment[i] = score[i]
                else:
                    score_fragment[i] = T.infinity(T.float32)
            for i in T.Parallel(32):
                if i < 16:
                    T.device_assert(T.isfinite(score_fragment[i]))
