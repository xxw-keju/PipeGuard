# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
LOG2E: float = 1.4426950408889634

def _resolve_kernel(**pipeguard_bindings):
    BV = pipeguard_bindings['BV']
    accum_dtype = pipeguard_bindings['accum_dtype']
    batch = pipeguard_bindings['batch']
    block_C = pipeguard_bindings['block_C']
    dim_aug = pipeguard_bindings['dim_aug']
    dim_k = pipeguard_bindings['dim_k']
    dim_v = pipeguard_bindings['dim_v']
    dtype = pipeguard_bindings['dtype']
    group_chunks = pipeguard_bindings['group_chunks']
    head = pipeguard_bindings['head']
    num_groups = pipeguard_bindings['num_groups']
    num_v_tiles = pipeguard_bindings['num_v_tiles']
    seq_len = pipeguard_bindings['seq_len']

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=256):

        @T.prim_func
        def group_transition_summary_bthd(k: T.Tensor([batch, seq_len, head, dim_k], dtype), g: T.Tensor([batch, seq_len, head], dtype), w: T.Tensor([batch, seq_len, head, dim_k], dtype), u: T.Tensor([batch, seq_len, head, dim_v], dtype), summary: T.Tensor([batch, head, num_groups, dim_k, dim_aug], dtype)):
            with T.Kernel(num_v_tiles * num_groups, batch, head, threads=threads) as (vgid, bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                vid = vgid % num_v_tiles
                gid = vgid // num_v_tiles
                v_offset = vid * BV
                start_chunk = gid * group_chunks
                for i, j in T.Parallel(dim_k, BV):
                    aug_col = v_offset + j
                    if aug_col < dim_v:
                        h_c[i, j] = 0.0
                    else:
                        a_col = aug_col - dim_v
                        if i == a_col:
                            h_c[i, j] = 1.0
                        else:
                            h_c[i, j] = 0.0
                for local_t in T.Pipelined(group_chunks, num_stages=num_stages):
                    t = start_chunk + local_t
                    base = t * block_C
                    for i, d in T.Parallel(block_C, dim_k):
                        k_c[i, d] = k[bid, base + i, hid, d]
                        w_c[i, d] = w[bid, base + i, hid, d]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, base + i, hid]
                    for i, j in T.Parallel(block_C, BV):
                        aug_col = v_offset + j
                        if aug_col < dim_v:
                            u_c[i, j] = u[bid, base + i, hid, aug_col]
                        else:
                            u_c[i, j] = 0.0
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * LOG2E)
                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2((g_c[block_C - 1] - g_c[n]) * LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_c[block_C - 1] * LOG2E)
                    T.gemm(k_c, v_new_c, h_next_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(h_next_frag, h_c)
                for i, j in T.Parallel(dim_k, BV):
                    summary[bid, hid, gid, i, v_offset + j] = h_c[i, j]
        return group_transition_summary_bthd
    return _func

def build_kernel(case):
    return _resolve_kernel(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
