    @T.prim_func
    def main(
        A: T.Tensor((64, 128), T.float4_e2m1fn),
    ):
        with T.Kernel(T.ceildiv(128, 64), T.ceildiv(64, 32), threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 64), T.float4_e2m1_unpacked)
            T.copy(A[by * 32, bx * 64], A_shared)
