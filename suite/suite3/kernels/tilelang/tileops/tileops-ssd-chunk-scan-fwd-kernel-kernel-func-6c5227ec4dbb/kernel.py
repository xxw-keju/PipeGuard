# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, num_chunks: int, chunk_len: int, n_heads: int, d_head: int, d_state: int, n_groups: int, dtype: str='float16', diagonal_microtile_size: int=32) -> Callable:
    accum_dtype = 'float'
    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    S = C * Q
    HEADS_PER_GROUP = H // G

    @tilelang.jit(out_idx=[-1])
    def kernel_func(block_l: int, block_p: int, block_n: int, block_s: int, threads: int, num_stages: int):
        x_shape = (B, S, H, P)
        cb_shape = (B, C, G, Q, Q)
        dA_shape = (B, H, C, Q)
        c_shape = (B, S, G, N)
        states_shape = (B, C, H, P, N)
        dt_shape = (B, H, C, Q)
        out_shape = (B, S, H, P)

        @T.prim_func
        def main(x: T.Tensor(x_shape, dtype), cb: T.Tensor(cb_shape, dtype), dA_cumsum: T.Tensor(dA_shape, accum_dtype), C_mat: T.Tensor(c_shape, dtype), prev_states: T.Tensor(states_shape, accum_dtype), dt: T.Tensor(dt_shape, dtype), out: T.Tensor(out_shape, accum_dtype)):
            with T.Kernel(T.ceildiv(Q, block_l) * T.ceildiv(P, block_p), B * C, H, threads=threads) as (blp, bc, bh):
                bl = blp // T.ceildiv(P, block_p)
                bp = blp % T.ceildiv(P, block_p)
                bz = bc // C
                bc_idx = bc % C
                bg = bh // HEADS_PER_GROUP
                chunk_start = bc_idx * Q
                l0 = bl * block_l
                p0 = bp * block_p
                acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(acc)
                hist_acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(hist_acc)
                c_tile = T.alloc_shared((block_l, block_n), dtype)
                state_tile = T.alloc_shared((block_n, block_p), dtype)
                T.annotate_layout({c_tile: tilelang.layout.make_swizzled_layout(c_tile), state_tile: tilelang.layout.make_swizzled_layout(state_tile)})
                for n_blk in T.Pipelined(T.ceildiv(N, block_n), num_stages=num_stages):
                    n0 = n_blk * block_n
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_abs = l0 + ll
                        n_abs = n0 + nn
                        safe_l = T.min(l_abs, Q - 1)
                        safe_n_c = T.min(n_abs, N - 1)
                        c_tile[ll, nn] = T.if_then_else(l_abs < Q and n_abs < N, C_mat[bz, chunk_start + safe_l, bg, safe_n_c], T.cast(T.float32(0.0), dtype))
                    for pp, nn in T.Parallel(block_p, block_n):
                        n_abs = n0 + nn
                        p_abs = p0 + pp
                        safe_n = T.min(n_abs, N - 1)
                        safe_p = T.min(p_abs, P - 1)
                        state_tile[nn, pp] = T.if_then_else(n_abs < N and p_abs < P, T.cast(prev_states[bz, bc_idx, bh, safe_p, safe_n], dtype), T.cast(T.float32(0.0), dtype))
                    T.gemm(c_tile, state_tile, hist_acc)
                dA_smem = T.alloc_shared((Q,), accum_dtype)
                dt_smem = T.alloc_shared((Q,), accum_dtype)
                for q in T.Parallel(Q):
                    dA_smem[q] = dA_cumsum[bz, bh, bc_idx, q]
                    dt_smem[q] = T.cast(dt[bz, bh, bc_idx, q], accum_dtype)
                T.sync_threads()
                exp_dA_l = T.alloc_shared((block_l,), accum_dtype)
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    safe_l_da = T.min(l_abs, Q - 1)
                    exp_dA_l[ll] = T.if_then_else(l_abs < Q, T.exp(dA_smem[safe_l_da]), T.float32(1.0))
                T.sync_threads()
                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if l_abs < Q and p_abs < P:
                        acc[ll, pp] += hist_acc[ll, pp] * exp_dA_l[ll]
                cb_tile = T.alloc_shared((block_l, block_s), dtype)
                x_tile = T.alloc_shared((block_s, block_p), dtype)
                exp_l = T.alloc_shared((block_l,), accum_dtype)
                exp_s = T.alloc_shared((block_s,), accum_dtype)
                lcb_cast = T.alloc_fragment((block_l, block_s), dtype)
                T.annotate_layout({cb_tile: tilelang.layout.make_swizzled_layout(cb_tile), x_tile: tilelang.layout.make_swizzled_layout(x_tile)})
                safe_l0 = T.min(l0, Q - 1)
                anchor = T.if_then_else(l0 < Q, dA_smem[safe_l0], T.float32(0.0))
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    safe_l_exp = T.min(l_abs, Q - 1)
                    dA_l_val = T.if_then_else(l_abs < Q, dA_smem[safe_l_exp], anchor)
                    exp_l[ll] = T.exp(dA_l_val - anchor)
                T.sync_threads()
                for s_blk in T.Pipelined(T.ceildiv(l0 + block_l, block_s), num_stages=num_stages):
                    s0 = s_blk * block_s
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        safe_l_cb = T.min(l_abs, Q - 1)
                        safe_s_cb = T.min(s_abs, Q - 1)
                        cb_tile[ll, ss] = T.if_then_else(l_abs < Q and s_abs < Q, cb[bz, bc_idx, bg, safe_l_cb, safe_s_cb], T.cast(T.float32(0.0), dtype))
                    for ss, pp in T.Parallel(block_s, block_p):
                        s_abs = s0 + ss
                        p_abs = p0 + pp
                        safe_s_x = T.min(s_abs, Q - 1)
                        safe_p_x = T.min(p_abs, P - 1)
                        x_tile[ss, pp] = T.if_then_else(s_abs < Q and p_abs < P, x[bz, chunk_start + safe_s_x, bh, safe_p_x], T.cast(T.float32(0.0), dtype))
                    if s0 + block_s <= l0:
                        for ss in T.Parallel(block_s):
                            s_abs = s0 + ss
                            safe_s_full = T.min(s_abs, Q - 1)
                            dA_s_val = T.if_then_else(s_abs < Q, dA_smem[safe_s_full], anchor)
                            dt_val = T.if_then_else(s_abs < Q, dt_smem[safe_s_full], T.float32(0.0))
                            exp_s[ss] = T.exp(anchor - dA_s_val) * dt_val
                        T.sync_threads()
                        for ll, ss in T.Parallel(block_l, block_s):
                            lcb_cast[ll, ss] = T.cast(T.cast(cb_tile[ll, ss], accum_dtype) * exp_l[ll] * exp_s[ss], dtype)
                    else:
                        T.sync_threads()
                        if diagonal_microtile_size == 32 and block_l == 64 and (block_s == 64):
                            M = 32
                            row_factor = T.alloc_shared((M,), accum_dtype)
                            col_factor = T.alloc_shared((M,), accum_dtype)
                            anchor_idx = T.min(l0 + M, Q - 1)
                            anchor = dA_smem[anchor_idx]
                            for i in T.Parallel(M):
                                safe_l = T.min(l0 + M + i, Q - 1)
                                row_factor[i] = T.exp(dA_smem[safe_l] - anchor)
                                safe_s = T.min(s0 + i, Q - 1)
                                col_factor[i] = T.exp(anchor - dA_smem[safe_s]) * dt_smem[safe_s]
                            T.sync_threads()
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + ll
                                s_abs = s0 + ss
                                valid = l_abs < Q and s_abs < Q and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = T.cast(cb_tile[ll, ss], accum_dtype) * exp_factor * dt_smem[safe_s]
                                lcb_cast[ll, ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))
                            for ll, ss in T.Parallel(M, M):
                                lcb_cast[ll, M + ss] = T.cast(T.float32(0.0), dtype)
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + ss
                                valid = l_abs < Q and s_abs < Q and (s_abs <= l_abs)
                                value = T.cast(cb_tile[M + ll, ss], accum_dtype) * row_factor[ll] * col_factor[ss]
                                lcb_cast[M + ll, ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + M + ss
                                valid = l_abs < Q and s_abs < Q and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = T.cast(cb_tile[M + ll, M + ss], accum_dtype) * exp_factor * dt_smem[safe_s]
                                lcb_cast[M + ll, M + ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))
                        else:
                            for ll, ss in T.Parallel(block_l, block_s):
                                l_abs = l0 + ll
                                s_abs = s0 + ss
                                valid = l_abs < Q and s_abs < Q and (s_abs <= l_abs)
                                safe_l_diag = T.min(l_abs, Q - 1)
                                safe_s_diag = T.min(s_abs, Q - 1)
                                dA_l_val = T.if_then_else(l_abs < Q, dA_smem[safe_l_diag], T.float32(0.0))
                                dA_s_val = T.if_then_else(s_abs < Q, dA_smem[safe_s_diag], T.float32(0.0))
                                dt_val = T.if_then_else(s_abs < Q, dt_smem[safe_s_diag], T.float32(0.0))
                                lcb_cast[ll, ss] = T.if_then_else(valid, T.cast(T.cast(cb_tile[ll, ss], accum_dtype) * T.exp(dA_l_val - dA_s_val) * dt_val, dtype), T.cast(T.float32(0.0), dtype))
                    T.gemm(lcb_cast, x_tile, acc)
                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if l_abs < Q and p_abs < P:
                        out[bz, chunk_start + l_abs, bh, p_abs] = acc[ll, pp]
        return main
    return kernel_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
