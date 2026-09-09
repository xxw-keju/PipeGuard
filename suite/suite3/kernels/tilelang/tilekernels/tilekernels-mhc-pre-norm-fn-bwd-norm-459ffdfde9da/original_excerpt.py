@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_norm_kernel(
        # Gradient of output
        out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        # Saved inputs
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Computed gradient of inputs
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, n_rms_group, threads=n_thr) as (pid_i, pid_k):
            sqrsum_frag = T.alloc_fragment(1, T.float32)
            sqrsum_frag[0] = sqrsum[pid_i, pid_k]
            rms_frag = T.alloc_fragment(1, T.float32)
            rms_frag[0] = T.rsqrt(sqrsum_frag[0] / rms_group_size + rms_eps)

            rms_grad_frag = T.alloc_reducer(1, T.float32, replication='all')
            T.clear(rms_grad_frag)
            for j in T.Parallel(mhc_mult3):
                out_mul_grad[pid_i, pid_k, j] = out_grad[pid_i, j] * rms_frag[0]
                rms_grad_frag[0] += out_grad[pid_i, j] * out_mul[pid_i, pid_k, j]
            T.finalize_reducer(rms_grad_frag)

            for kk in T.Parallel(1):
                sqrsum_grad[pid_i, pid_k + kk] = rms_grad_frag[kk] * rms_frag[kk] / (sqrsum_frag[kk] + rms_eps * rms_group_size) / -2

    return _mhc_pre_norm_fn_bwd_norm_kernel
