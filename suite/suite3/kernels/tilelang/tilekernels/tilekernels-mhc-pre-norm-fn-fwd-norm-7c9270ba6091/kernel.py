# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/norm_fn_kernel.py#L126-L165
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_norm_kernel(
        out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
            rms = T.alloc_fragment(1, T.float32)
            out_l = T.alloc_fragment(mhc_mult3, T.float32)
            out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(out_l)
            for k in T.serial(n_rms_group):
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += sqrsum_splitted[i_split, pid, k]
                if T.get_thread_binding() == 0:
                    sqrsum[pid, k] = rms[0]
                rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    out_l0[j] = 0
                    for i_split in T.serial(n_splits):
                        out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                    out_l[j] += out_l0[j] * rms[0]
                T.copy(out_l0, out_mul[pid, k, :])
            T.copy(out_l[:], out[pid, :])

    return _mhc_pre_norm_fn_fwd_norm_kernel


def _resolve_kernel(**pipeguard_bindings):
    return _mhc_pre_norm_fn_fwd_norm


def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel()
    return factory(**case["build"]["factory_arguments"])
