# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str='float16') -> Callable:
    """Split-K variant of the 2-consumer (cooperative) GEMM mainloop (NT).

    For small-M shapes whose natural (M, N) grid underfills the GPU with a
    single K-slice, this slices K across grid-z CTAs. Each CTA runs the coop2
    mainloop (1 producer + 2 math WGs, split-A / shared-B, ``block_m`` fixed at
    128 = two 64-row consumers) over its K-slice and writes an fp32 partial tile
    into ``w[split_k, m, n]``; ``_splitk_reduce_kernel`` then sums the slices and
    casts to the storage dtype.

    The 2-consumer mainloop is the more WGMMA-efficient of the two, so it takes
    the large-K decode shapes, whose slices amortize the reduce round-trip;
    dispatch keeps short-K shapes on ``_gemm_splitk_kernel``.

    NT only. ``split_k`` must divide the K-tile count evenly.

    Args:
        m: Rows of ``A`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Must be ``False`` (NT layout).
        trans_b: Must be ``True`` (NT layout).
        dtype: Activation / weight dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_n, block_k,
        num_stages, split_k)`` returns the compiled ``prim_func`` producing the
        fp32 workspace ``w[split_k, m, n]``.
    """
    if trans_a or not trans_b:
        raise ValueError('_gemm_coop2_splitk_kernel is NT-only (trans_a=False, trans_b=True)')
    accum_dtype = 'float'
    block_m = 128

    @tilelang.jit(out_idx=[-1], pass_configs={'tl.disable_warp_specialized': True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_coop2_splitk_func(block_n: int=64, block_k: int=128, num_stages: int=4, split_k: int=4) -> Callable:
        half_m = block_m // 2
        nr = half_m * block_n // 128
        k_iters_total = T.ceildiv(k, block_k)
        if k_iters_total % split_k != 0:
            raise ValueError(f'split_k={split_k} must divide the K-tile count evenly (k={k}, block_k={block_k} -> {k_iters_total} tiles)')
        k_slice = k_iters_total // split_k

        @T.prim_func
        def _gemm_coop2_splitk_main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k), dtype), w: T.Tensor((split_k, m, n), accum_dtype)) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), split_k, threads=384) as (bx, by, bz):
                a_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                a_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                b_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                c_local_0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                c_local_1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                T.annotate_layout({a_smem_top: tilelang.layout.make_swizzled_layout(a_smem_top), a_smem_bot: tilelang.layout.make_swizzled_layout(a_smem_bot), b_smem: tilelang.layout.make_swizzled_layout(b_smem)})
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                ps0 = T.alloc_local((1,), 'int32')
                ps1 = T.alloc_local((1,), 'int32')
                m_start = by * block_m
                n_start = bx * block_n
                ki_base = bz * k_slice
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        ks = (ki_base + ki) * block_k
                        T.barrier_wait(ab_empty[slot], ki // num_stages % 2 ^ 1)
                        T.tma_copy(a[m_start:m_start + half_m, ks:ks + block_k], a_smem_top[slot, :, :], barrier=ab_full[slot])
                        T.tma_copy(a[m_start + half_m:m_start + block_m, ks:ks + block_k], a_smem_bot[slot, :, :], barrier=ab_full[slot])
                        T.tma_copy(b[n_start:n_start + block_n, ks:ks + block_k], b_smem[slot, :, :], barrier=ab_full[slot])
                        T.barrier_arrive(ab_full[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], ki // num_stages % 2)
                        T.wgmma_gemm(a_smem_top[slot, :, :], b_smem[slot, :, :], c_local_0, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps0[0]])
                        ps0[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps0[0]])
                    T.warpgroup_fence_operand(c_local_0, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            w[bz, m_start + i, n_start + j] = c_local_0[i, j]
                else:
                    T.inc_max_nreg(240)
                    for ki in T.Pipelined(k_slice, num_stages=0):
                        slot = ki % num_stages
                        T.barrier_wait(ab_full[slot], ki // num_stages % 2)
                        T.wgmma_gemm(a_smem_bot[slot, :, :], b_smem[slot, :, :], c_local_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[ps1[0]])
                        ps1[0] = slot
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[ps1[0]])
                    T.warpgroup_fence_operand(c_local_1, num_regs=nr)
                    for i, j in T.Parallel(half_m, block_n):
                        if m_start + half_m + i < m and n_start + j < n:
                            w[bz, m_start + half_m + i, n_start + j] = c_local_1[i, j]
        return _gemm_coop2_splitk_main
    return _gemm_coop2_splitk_func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    return _kernel_factory(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
