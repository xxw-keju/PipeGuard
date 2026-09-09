    @T.prim_func
    def main(A: T.Tensor((256, 512), T.float4_e2m1fn)):
        with T.Kernel(1, threads=32):
            T.evaluate(A[0, 0])
