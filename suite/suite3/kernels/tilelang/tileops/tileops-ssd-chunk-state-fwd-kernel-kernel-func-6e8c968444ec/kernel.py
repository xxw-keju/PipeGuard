# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, num_chunks: int, chunk_len: int, n_heads: int, d_head: int, d_state: int, n_groups: int, has_seq_idx: bool=False, dtype: str='float16', dt_dtype: str='float32') -> Callable:
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
    def kernel_func(block_n: int, block_p: int, block_l: int, threads: int, num_stages: int):

        @T.prim_func
        def main(x: T.Tensor((B, S, H, P), dtype), Bmat: T.Tensor((B, S, G, N), dtype), dt: T.Tensor((B, H, C, Q), dt_dtype), dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype), seq_idx: T.Tensor((B, S), 'int32'), out: T.Tensor((B, C, H, P, N), accum_dtype)):
            with T.Kernel(B * H * C, T.ceildiv(P, block_p), T.ceildiv(N, block_n), threads=threads) as (bhc, bp, bn):
                bz = bhc // (C * H)
                bc = bhc % (C * H) // H
                bh = bhc % H
                n0 = bn * block_n
                p0 = bp * block_p
                bg = bh // HEADS_PER_GROUP
                chunk_start = bc * Q
                acc = T.alloc_fragment((block_p, block_n), accum_dtype)
                T.clear(acc)
                dA_end = dA_cumsum[bz, bh, bc, Q - 1]
                seq_end = seq_idx[bz, chunk_start + Q - 1] if has_seq_idx else T.int32(0)
                w_tile = T.alloc_shared((block_l,), accum_dtype)
                x_scaled = T.alloc_shared((block_l, block_p), dtype)
                b_tile = T.alloc_shared((block_l, block_n), dtype)
                for l_blk in T.Pipelined(T.ceildiv(Q, block_l), num_stages=num_stages):
                    l0 = l_blk * block_l
                    for ll in T.Parallel(block_l):
                        l_idx = l0 + ll
                        dA_l = T.if_then_else(l_idx < Q, dA_cumsum[bz, bh, bc, l_idx], T.float32(0.0))
                        dt_l = T.if_then_else(l_idx < Q, T.cast(dt[bz, bh, bc, l_idx], accum_dtype), T.float32(0.0))
                        if has_seq_idx:
                            same_seq = T.if_then_else(l_idx < Q, seq_end >= T.int32(0) and seq_idx[bz, chunk_start + l_idx] == seq_end, T.bool(False))
                            w_tile[ll] = T.if_then_else(same_seq, T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l, T.float32(0.0))
                        else:
                            w_tile[ll] = T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l
                    T.sync_threads()
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        x_val = T.if_then_else(l_idx < Q and p_idx < P, T.cast(x[bz, chunk_start + l_idx, bh, p_idx], accum_dtype), T.float32(0.0))
                        x_scaled[ll, pp] = T.cast(x_val * w_tile[ll], dtype)
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_idx = l0 + ll
                        n_idx = n0 + nn
                        b_tile[ll, nn] = T.if_then_else(l_idx < Q and n_idx < N, Bmat[bz, chunk_start + l_idx, bg, n_idx], T.cast(T.float32(0.0), dtype))
                    T.gemm(x_scaled, b_tile, acc, transpose_A=True)
                for pp, nn in T.Parallel(block_p, block_n):
                    p_idx = p0 + pp
                    n_idx = n0 + nn
                    if p_idx < P and n_idx < N:
                        out[bz, bc, bh, p_idx, n_idx] = acc[pp, nn]
        return main
    return kernel_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
