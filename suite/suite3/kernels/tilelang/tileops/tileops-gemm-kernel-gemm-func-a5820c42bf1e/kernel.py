# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(m: int, n: int, k: int, trans_a: bool, trans_b: bool, dtype: str='float16', traced: bool=False, *, sm_count: int) -> Callable:
    """Hand-written warp-specialized GEMM ``C = op(A) @ op(B)`` for Hopper (SM90).

    One producer warpgroup (128 threads) issues TMA loads into a double-buffered
    SMEM ring; one consumer warpgroup (128 threads) runs the WGMMA and accumulates
    over K. All four layouts are covered by ``trans_a`` / ``trans_b`` (forwarded to
    the WGMMA transpose flags): ``A`` is $[M \\times K]$ (or $[K \\times M]$ transposed), ``B``
    is $[K \\times N]$ (or $[N \\times K]$ transposed), ``C`` is $[M \\times N]$. fp16 / bf16 inputs,
    fp32 accumulation. The auto warp-specialization pass is disabled so it does not
    fire on top of this manual layout.

    Operands must satisfy TMA's innermost-dimension alignment, which
    ``_tma_misalignment`` states and ``GemmKernel`` refuses on.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        trans_a: Whether ``A`` is stored transposed ($[K \\times M]$).
        trans_b: Whether ``B`` is stored transposed ($[N \\times K]$).
        dtype: Activation / weight dtype string (``"float16"`` or ``"bfloat16"``).
        traced: Build with in-kernel timeline markers materialized (``True``) or
            stripped to zero cost (``False``). **Part of the cache key**: traced
            and untraced builds are distinct cached kernels, so flipping the
            process trace switch never returns a stale variant. Callers pass
            ``trace.enabled`` explicitly rather than letting the build read the
            global switch.
        sm_count: Device SM count, deciding the multi-wave TMA-store epilogue
            gate. Part of the cache key so a kernel built for one GPU is never
            reused on another.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, panel_size)`` returns the compiled ``prim_func``.
        When ``traced``
        it materializes the markers and appends a trailing ``slots`` output (so
        ``out_idx`` returns ``(C, slots)``); otherwise ``C`` is the lone output.
    """
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={'tl.disable_warp_specialized': True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_func(block_m: int=128, block_n: int=128, block_k: int=64, num_stages: int=3, panel_size: int=10) -> Callable:
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)
        grid_size = -(-n // block_n) * -(-m // block_m)
        tma_epilogue = n * 2 % 16 == 0 and grid_size > sm_count

        @T.prim_func
        def _gemm_main(a: T.Tensor((k, m) if trans_a else (m, k), dtype), b: T.Tensor((n, k) if trans_b else (k, n), dtype), c: T.Tensor((m, n), dtype)) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                T.use_swizzle(panel_size, enable=panel_size > 0)
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                if tma_epilogue:
                    c_smem = T.alloc_shared((block_m, block_n), dtype)
                    T.annotate_layout({a_smem: tilelang.layout.make_swizzled_layout(a_smem), b_smem: tilelang.layout.make_swizzled_layout(b_smem), c_smem: tilelang.layout.make_swizzled_layout(c_smem)})
                else:
                    T.annotate_layout({a_smem: tilelang.layout.make_swizzled_layout(a_smem), b_smem: tilelang.layout.make_swizzled_layout(b_smem)})
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)
                m_start = by * block_m
                n_start = bx * block_n
                ps = T.alloc_local((1,), 'int32')
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        k_start = ki * block_k
                        T.barrier_wait(ab_empty[slot], phase ^ 1)
                        if trans_a:
                            T.tma_copy(a[k_start:k_start + block_k, m_start:m_start + block_m], a_smem[slot, :, :], barrier=ab_full[slot])
                        else:
                            T.tma_copy(a[m_start:m_start + block_m, k_start:k_start + block_k], a_smem[slot, :, :], barrier=ab_full[slot])
                        if trans_b:
                            T.tma_copy(b[n_start:n_start + block_n, k_start:k_start + block_k], b_smem[slot, :, :], barrier=ab_full[slot])
                        else:
                            T.tma_copy(b[k_start:k_start + block_k, n_start:n_start + block_n], b_smem[slot, :, :], barrier=ab_full[slot])
                        T.barrier_arrive(ab_full[slot])
                else:
                    T.inc_max_nreg(240)
                    num_accum_regs = block_m * block_n // 128
                    for ki in T.serial(k_iters):
                        slot = ki % num_stages
                        phase = ki // num_stages % 2
                        T.barrier_wait(ab_full[slot], phase)
                        T.wgmma_gemm(a_smem[slot, :, :], b_smem[slot, :, :], c_local, transpose_A=trans_a, transpose_B=trans_b, policy=T.GemmWarpPolicy.FullRow, clear_accum=ki == 0)
                        if ki > 0:
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                            T.barrier_arrive(ab_empty[ps[0]])
                        ps[0] = slot
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(c_local, num_regs=num_accum_regs)
                    T.barrier_arrive(ab_empty[ps[0]])
                    if tma_epilogue:
                        T.copy(c_local, c_smem)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.copy(c_smem, c[m_start, n_start])
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            if m_start + i < m and n_start + j < n:
                                c[m_start + i, n_start + j] = c_local[i, j]
        return _gemm_main
    return _gemm_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
