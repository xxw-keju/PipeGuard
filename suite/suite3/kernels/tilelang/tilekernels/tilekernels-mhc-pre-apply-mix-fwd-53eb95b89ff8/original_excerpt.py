@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_apply_mix_fwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic('n')
    h = hidden
    mhc = mhc_mult

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def _mhc_pre_apply_mix_fwd_kernel(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                os = T.alloc_shared(h_blk, T.bfloat16)
                ol = T.alloc_fragment(h_blk, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc):
                    for i1_h in T.Parallel(h_blk):
                        ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

                T.copy(ol, os, disable_tma=True)
                T.copy(os, o[pid_n, i0_h * h_blk], disable_tma=True)

    return _mhc_pre_apply_mix_fwd_kernel
