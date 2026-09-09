# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, m: int, n: int, k: int, dtype: str, out_dtype: str, sm_count: int) -> Callable:
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True, 'tl.disable_warp_specialized': True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _bmm_fp8_persistent_ws_func(block_m: int=128, block_n: int=128, block_k: int=128, num_stages: int=3, threads: int=384, group_size_m: int=8) -> Callable:
        assert threads == 384, f'3-WG persistent WS BMM requires threads=384 (1 producer + 2 consumer WGs); got threads={threads}'
        assert block_m % 2 == 0 and block_m // 2 >= 64, f'cooperative template needs block_m>=128 and even; got {block_m}'
        assert m % block_m == 0 and n % block_n == 0 and (k % block_k == 0), f'WS variant requires aligned tile; got m={m}, n={n}, k={k}, block=({block_m},{block_n},{block_k})'
        half_m = block_m // 2
        num_pid_m = m // block_m
        num_pid_n = n // block_n
        k_iters = k // block_k
        assert k_iters >= 2, f'WS variant needs k_iters>=2 for the 1-deep WGMMA pipeline; got k={k}, block_k={block_k}, k_iters={k_iters}'
        total_tiles = batch * num_pid_m * num_pid_n
        grid_cta = min(sm_count, total_tiles)
        max_waves = (total_tiles + grid_cta - 1) // grid_cta + 1
        scale_a_shape = (1,)
        scale_b_shape = (1,)

        @T.macro
        def _swizzle_decode(mn_id, swz_m, swz_n):
            _gin = T.int32(group_size_m * num_pid_n)
            _fpm = mn_id // _gin * T.int32(group_size_m)
            if T.int32(num_pid_m) - _fpm >= T.int32(group_size_m):
                swz_m[0] = _fpm + mn_id % _gin % T.int32(group_size_m)
                swz_n[0] = mn_id % _gin // T.int32(group_size_m)
            else:
                _gsm = T.int32(num_pid_m) - _fpm
                swz_m[0] = _fpm + mn_id % _gin % _gsm
                swz_n[0] = mn_id % _gin // _gsm

        @T.prim_func
        def _bmm_fp8_persistent_ws_main(a: T.Tensor((batch, m, k), dtype), b: T.Tensor((batch, n, k), dtype), scale_a: T.Tensor(scale_a_shape, 'float32'), scale_b: T.Tensor(scale_b_shape, 'float32'), c: T.Tensor((batch, m, n), out_dtype)) -> None:
            with T.Kernel(grid_cta, threads=threads) as (pid,):
                A_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                A_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                B_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
                C_local_wg0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                C_local_wg1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                C_local_cast_wg0 = T.alloc_fragment((half_m, block_n), out_dtype)
                C_local_cast_wg1 = T.alloc_fragment((half_m, block_n), out_dtype)
                C_shared_wg0 = T.alloc_shared((half_m, block_n), out_dtype)
                C_shared_wg1 = T.alloc_shared((half_m, block_n), out_dtype)
                bs = T.alloc_local((1,), 'int32')
                ms = T.alloc_local((1,), 'int32')
                ns_ = T.alloc_local((1,), 'int32')
                ps0 = T.alloc_local((1,), 'int32')
                ps1 = T.alloc_local((1,), 'int32')
                swz_m = T.alloc_local((1,), 'int32')
                swz_n = T.alloc_local((1,), 'int32')
                T.annotate_layout({A_smem_top: tilelang.layout.make_swizzled_layout(A_smem_top), A_smem_bot: tilelang.layout.make_swizzled_layout(A_smem_bot), B_smem: tilelang.layout.make_swizzled_layout(B_smem), C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0), C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1)})
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)
                gi_prod = T.alloc_var('int32', init=0)
                gi_cons_0 = T.alloc_var('int32', init=0)
                gi_cons_1 = T.alloc_var('int32', init=0)
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid
                        if flat_id < T.int32(total_tiles):
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)
                            for kk in T.Pipelined(k_iters, num_stages=0):
                                k_start = kk * block_k
                                slot = gi_prod % num_stages
                                T.barrier_wait(ab_empty[slot], gi_prod // num_stages & 1 ^ 1)
                                T.tma_copy(a[bs[0], ms[0]:ms[0] + half_m, k_start:k_start + block_k], A_smem_top[slot, :, :], barrier=ab_full[slot])
                                T.tma_copy(a[bs[0], ms[0] + half_m:ms[0] + block_m, k_start:k_start + block_k], A_smem_bot[slot, :, :], barrier=ab_full[slot])
                                T.tma_copy(b[bs[0], ns_[0]:ns_[0] + block_n, k_start:k_start + block_k], B_smem[slot, :, :], barrier=ab_full[slot])
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid
                        if flat_id < T.int32(total_tiles):
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)
                            for kk in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_0 % num_stages
                                T.barrier_wait(ab_full[slot], gi_cons_0 // num_stages & 1)
                                T.wgmma_gemm(A_smem_top[slot, :, :], B_smem[slot, :, :], C_local_wg0, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=kk == 0)
                                if kk > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps0[0]])
                                ps0[0] = slot
                                gi_cons_0 = gi_cons_0 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                            for i, j in T.Parallel(half_m, block_n):
                                C_local_wg0[i, j] = C_local_wg0[i, j] * scale_a[0] * scale_b[0]
                            T.copy(C_local_wg0, C_local_cast_wg0)
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_shared_wg0, c[bs[0], ms[0], ns_[0]])
                else:
                    T.inc_max_nreg(240)
                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid
                        if flat_id < T.int32(total_tiles):
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)
                            for kk in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_1 % num_stages
                                T.barrier_wait(ab_full[slot], gi_cons_1 // num_stages & 1)
                                T.wgmma_gemm(A_smem_bot[slot, :, :], B_smem[slot, :, :], C_local_wg1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=kk == 0)
                                if kk > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps1[0]])
                                ps1[0] = slot
                                gi_cons_1 = gi_cons_1 + 1
                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                            for i, j in T.Parallel(half_m, block_n):
                                C_local_wg1[i, j] = C_local_wg1[i, j] * scale_a[0] * scale_b[0]
                            T.copy(C_local_wg1, C_local_cast_wg1)
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_shared_wg1, c[bs[0], ms[0] + half_m, ns_[0]])
        return _bmm_fp8_persistent_ws_main
    return _bmm_fp8_persistent_ws_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
