# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T
CONV_SWIZZLE_PANEL = 2

def _resolve_kernel(**pipeguard_bindings):
    accum_dtype = pipeguard_bindings['accum_dtype']
    c_in = pipeguard_bindings['c_in']
    c_out = pipeguard_bindings['c_out']
    dilation = pipeguard_bindings['dilation']
    dtype = pipeguard_bindings['dtype']
    h = pipeguard_bindings['h']
    has_bias = pipeguard_bindings['has_bias']
    k_total = pipeguard_bindings['k_total']
    kernel_size = pipeguard_bindings['kernel_size']
    n = pipeguard_bindings['n']
    out_h = pipeguard_bindings['out_h']
    out_hw = pipeguard_bindings['out_hw']
    out_w = pipeguard_bindings['out_w']
    pad = pipeguard_bindings['pad']
    stride = pipeguard_bindings['stride']
    w = pipeguard_bindings['w']

    @tilelang.jit(out_idx=[4], compile_flags=['-O3', '-DENABLE_BF16'], pass_configs={'tl.enable_async_copy': False})
    def _conv2d_symmetric_func(block_m: int, block_n: int, block_k: int, num_stages: int, threads: int, enable_rasterization: bool):
        staging_threads = 256
        channel_block = 32
        x_lanes, x_spatial_block = (8, 32)
        w_lanes, w_spatial_block = (16, 16)
        assert x_spatial_block * x_lanes == staging_threads and w_spatial_block * w_lanes == staging_threads, "each operand's tile must occupy the threads the staging kernel launches"
        assert channel_block % x_lanes == 0 and channel_block % w_lanes == 0, "a thread's channel stride must divide the channel block it walks"
        channel_blocks = -(-c_in // channel_block)
        x_spatial_blocks = -(-(h * w) // x_spatial_block)
        w_spatial_blocks = -(-(kernel_size * kernel_size) // w_spatial_block)
        x_blocks = x_spatial_blocks * channel_blocks * n
        w_blocks = w_spatial_blocks * channel_blocks * c_out

        @T.macro
        def plane_to_channel_last(src: T.Tensor, dst: T.Tensor, block: T.int32, spatial_size: int, spatial_block: int, spatial_blocks: int, channel_lanes: int):
            """One block's share of ``src[p, c, s] -> dst[p, s, c]``."""
            spatial_base = block % spatial_blocks * spatial_block
            channel_base = block // spatial_blocks % channel_blocks * channel_block
            plane = block // (spatial_blocks * channel_blocks)
            for spatial_inner, channel_lane in T.Parallel(spatial_block, channel_lanes):
                spatial = spatial_base + spatial_inner
                for channel_offset in T.serial(channel_block // channel_lanes):
                    c = channel_base + channel_offset * channel_lanes + channel_lane
                    if (spatial < spatial_size) & (c < c_in):
                        dst[plane, spatial, c] = src[plane, c, spatial]

        @T.macro
        def stage_operands(x, weight, x_nhwc, weight_krsc):
            with T.Kernel(x_blocks + w_blocks, threads=staging_threads) as block:
                if block < x_blocks:
                    plane_to_channel_last(T.Tensor((n, c_in, h * w), dtype, x.data), T.Tensor((n, h * w, c_in), dtype, x_nhwc.data), block, spatial_size=h * w, spatial_block=x_spatial_block, spatial_blocks=x_spatial_blocks, channel_lanes=x_lanes)
                else:
                    plane_to_channel_last(T.Tensor((c_out, c_in, kernel_size * kernel_size), dtype, weight.data), T.Tensor((c_out, kernel_size * kernel_size, c_in), dtype, weight_krsc.data), block - x_blocks, spatial_size=kernel_size * kernel_size, spatial_block=w_spatial_block, spatial_blocks=w_spatial_blocks, channel_lanes=w_lanes)

        @T.macro
        def conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out, bias):
            with T.Kernel(T.ceildiv(c_out, block_n), T.ceildiv(n * out_h * out_w, block_m), threads=threads) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_n, block_m), dtype)
                weight_flat = T.Tensor((c_out, k_total), dtype, weight_krsc.data)
                out_nchw = T.Tensor((n, c_out, out_hw), dtype, out.data)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.im2col(x_nhwc, data_shared, by, k_iter, kernel_size, stride, dilation, pad)
                    T.copy(weight_flat[bx * block_n, k_iter * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[j, i] = T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype)
                    else:
                        out_shared[j, i] = T.cast(out_local[i, j], dtype)
                m_base = by * block_m
                T.copy(out_shared, out_nchw[m_base // out_hw, bx * block_n, m_base % out_hw])

        @T.macro
        def _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, bias):
            stage_operands(x, weight, x_nhwc, weight_krsc)
            conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out, bias)
        if has_bias:

            @T.prim_func
            def _conv2d_symmetric_bias_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype), x_nhwc: T.Tensor((n, h, w, c_in), dtype), weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype), out: T.Tensor((n, c_out, out_h, out_w), dtype), bias: T.Tensor((c_out,), dtype)):
                _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, bias)
            return _conv2d_symmetric_bias_main

        @T.prim_func
        def _conv2d_symmetric_main(x: T.Tensor((n, c_in, h, w), dtype), weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype), x_nhwc: T.Tensor((n, h, w, c_in), dtype), weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype), out: T.Tensor((n, c_out, out_h, out_w), dtype)):
            _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, None)
        return _conv2d_symmetric_main
    return _conv2d_symmetric_func

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('factory_arguments', {}))
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
