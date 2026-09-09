# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
CONV_SWIZZLE_PANEL = 2

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    c_in = pipeguard_bindings['c_in']
    c_in_g = pipeguard_bindings['c_in_g']
    c_out = pipeguard_bindings['c_out']
    c_out_g = pipeguard_bindings['c_out_g']
    dilation_l = pipeguard_bindings['dilation_l']
    dtype = pipeguard_bindings['dtype']
    groups = pipeguard_bindings['groups']
    has_bias = pipeguard_bindings['has_bias']
    k_total = pipeguard_bindings['k_total']
    kernel_l = pipeguard_bindings['kernel_l']
    l_in = pipeguard_bindings['l_in']
    n = pipeguard_bindings['n']
    out_l = pipeguard_bindings['out_l']
    pad_left = pipeguard_bindings['pad_left']
    stride_l = pipeguard_bindings['stride_l']

    @tilelang.jit(out_idx=[2], compile_flags=['-O3', '-DENABLE_BF16'])
    def _conv1d_group_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):

        @T.macro
        def _conv1d_group_body(x, weight, out, bias):
            with T.Kernel(T.ceildiv(out_l, block_n), T.ceildiv(c_out_g, block_m), n * groups, threads=threads) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                batch_id = bz // groups
                group_id = bz % groups
                oc_base = group_id * c_out_g + by * block_m
                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, k in T.Parallel(block_m, block_k):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        k_idx = k_iter * block_k + k
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        weight_shared[i, k] = T.if_then_else((oc_g < c_out_g) & (k_idx < k_total), weight[oc, ci_g, kw], T.cast(0.0, dtype))
                    for k, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + k
                        ol = bx * block_n + j
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        il = ol * stride_l + kw * dilation_l - pad_left
                        data_shared[k, j] = T.if_then_else((k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in), x[batch_id, group_id * c_in_g + ci_g, il], T.cast(0.0, dtype))
                    T.gemm(weight_shared, data_shared, out_local)
                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else((oc_g < c_out_g) & (ol < out_l), T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype), T.cast(0.0, dtype))
                    else:
                        out_shared[i, j] = T.if_then_else((oc_g < c_out_g) & (ol < out_l), T.cast(out_local[i, j], dtype), T.cast(0.0, dtype))
                if c_out_g % block_m == 0:
                    T.copy(out_shared, out[batch_id, oc_base, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        ol = bx * block_n + j
                        if oc_g < c_out_g and ol < out_l:
                            out[batch_id, oc, ol] = out_shared[i, j]
        if has_bias:

            @T.prim_func
            def _conv1d_group_bias_main(x: T.Tensor((n, c_in, l_in), dtype), weight: T.Tensor((c_out, c_in_g, kernel_l), dtype), out: T.Tensor((n, c_out, out_l), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv1d_group_body(x, weight, out, bias)
            return _conv1d_group_bias_main

        @T.prim_func
        def _conv1d_group_main(x: T.Tensor((n, c_in, l_in), dtype), weight: T.Tensor((c_out, c_in_g, kernel_l), dtype), out: T.Tensor((n, c_out, out_l), dtype)):
            _conv1d_group_body(x, weight, out, None)
        return _conv1d_group_main
    return _conv1d_group_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
