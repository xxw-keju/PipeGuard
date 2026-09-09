@tilelang.jit
def _mhc_fn_normw_merge_fwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_fwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return _mhc_fn_normw_merge_fwd_
