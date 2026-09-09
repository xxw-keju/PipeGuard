    @T.prim_func
    def main(A: T.Tensor((128), T.int32)):
        with T.ClusterKernel(128, threads=128, cluster_dims=(4, 1, 1)):
            mbar = T.alloc_cluster_barrier([256])
            T.cluster_sync()
            T.mbarrier_arrive(mbar, 0)
            if T.block_rank_in_cluster() == 0:
                T.mbarrier_wait_parity(mbar, 0)
            T.cluster_sync()
