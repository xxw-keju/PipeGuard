    @tilelang.jit(
        out_idx=[5],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _conv3d_ndhwc_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def transpose_spatial_channel(
            src: T.Tensor,
            dst: T.Tensor,
            batch_size: int,
            spatial_size: int,
            channel_size: int,
            hw_size: int,
            width: int,
            spatial_block: int,
            channel_block: int,
            channel_lanes: int,
            channel_fastest: bool,
            is_nchw_to_nhwc: bool,
        ):
            assert spatial_block * channel_lanes == 256, (
                "spatial_block * channel_lanes must equal 256"
            )
            assert channel_block % channel_lanes == 0, "channel_lanes must divide channel_block"
            if not channel_fastest:
                assert channel_block * spatial_block == 256, (
                    "channel_block * spatial_block must equal 256 when channel_fastest=False"
                )

            with T.Kernel(
                T.ceildiv(spatial_size, spatial_block),
                T.ceildiv(channel_size, channel_block),
                batch_size,
                threads=256,
            ) as (bx, by, bz):
                if channel_fastest:
                    values_per_thread = channel_block // channel_lanes
                    for spatial_inner, channel_lane in T.Parallel(spatial_block, channel_lanes):
                        spatial = bx * spatial_block + spatial_inner
                        d_idx = spatial // hw_size
                        rem = spatial - d_idx * hw_size
                        h_idx = rem // width
                        w_idx = rem - h_idx * width
                        channel_base = by * channel_block
                        for channel_offset in T.serial(values_per_thread):
                            c = channel_base + channel_offset * channel_lanes + channel_lane
                            if (spatial < spatial_size) & (c < channel_size):
                                if is_nchw_to_nhwc:
                                    dst[bz, d_idx, h_idx, w_idx, c] = src[
                                        bz, c, d_idx, h_idx, w_idx
                                    ]
                                else:
                                    dst[bz, c, d_idx, h_idx, w_idx] = src[
                                        bz, d_idx, h_idx, w_idx, c
                                    ]
                else:
                    for channel_inner, spatial_inner in T.Parallel(channel_block, spatial_block):
                        spatial = bx * spatial_block + spatial_inner
                        d_idx = spatial // hw_size
                        rem = spatial - d_idx * hw_size
                        h_idx = rem // width
                        w_idx = rem - h_idx * width
                        c = by * channel_block + channel_inner
                        if (spatial < spatial_size) & (c < channel_size):
                            if is_nchw_to_nhwc:
                                dst[bz, d_idx, h_idx, w_idx, c] = src[bz, c, d_idx, h_idx, w_idx]
                            else:
                                dst[bz, c, d_idx, h_idx, w_idx] = src[bz, d_idx, h_idx, w_idx, c]

        @T.macro
        def conv_ndhwc_implicit_gemm(x_ndhwc, weight_kdrsc, out_ndhwc, bias):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_dhw, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight_kdrsc.data)
                out_flat = T.Tensor((n * out_dhw, c_out), dtype, out_ndhwc.data)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        m = by * block_m + i
                        k = k_iter * block_k + j
                        c = k % c_in
                        tap = k // c_in
                        kw = tap % kernel_w
                        kh = (tap // kernel_w) % kernel_h
                        kd = tap // (kernel_h * kernel_w)
                        spatial = m % out_dhw
                        od = spatial // (out_h * out_w)
                        oh = (spatial // out_w) % out_h
                        ow = spatial % out_w
                        id_ = od * stride_d + kd * dilation_d - pad_d
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        in_bound = (
                            (m < n * out_dhw)
                            & (id_ >= 0)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (id_ < d)
                            & (ih < h)
                            & (iw < w)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x_ndhwc[m // out_dhw, id_, ih, iw, c],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[bx * block_n, k_iter * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    spatial_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (spatial_idx < n * out_dhw) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (spatial_idx < n * out_dhw) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[by * block_m, bx * block_n])

        @T.macro
        def _conv3d_ndhwc_body(x, weight, x_ndhwc, weight_kdrsc, out_ndhwc, out, bias):
            transpose_spatial_channel(
                x,
                x_ndhwc,
                batch_size=n,
                spatial_size=d * h * w,
                channel_size=c_in,
                hw_size=h * w,
                width=w,
                spatial_block=32,
                channel_block=32,
                channel_lanes=8,
                channel_fastest=True,
                is_nchw_to_nhwc=True,
            )
            transpose_spatial_channel(
                weight,
                weight_kdrsc,
                batch_size=c_out,
                spatial_size=kernel_d * kernel_h * kernel_w,
                channel_size=c_in,
                hw_size=kernel_h * kernel_w,
                width=kernel_w,
                spatial_block=16,
                channel_block=32,
                channel_lanes=16,
                channel_fastest=True,
                is_nchw_to_nhwc=True,
            )
            conv_ndhwc_implicit_gemm(x_ndhwc, weight_kdrsc, out_ndhwc, bias)
            transpose_spatial_channel(
                out_ndhwc,
                out,
                batch_size=n,
                spatial_size=out_dhw,
                channel_size=c_out,
                hw_size=out_h * out_w,
                width=out_w,
                spatial_block=128,
                channel_block=2,
                channel_lanes=2,
                channel_fastest=False,
                is_nchw_to_nhwc=False,
            )

        if has_bias:

            @T.prim_func
            def _conv3d_ndhwc_bias_main(
                x: T.Tensor((n, c_in, d, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
                x_ndhwc: T.Tensor((n, d, h, w, c_in), dtype),  # type: ignore
                weight_kdrsc: T.Tensor((c_out, kernel_d, kernel_h, kernel_w, c_in), dtype),  # type: ignore
                out_ndhwc: T.Tensor((n, out_d, out_h, out_w, c_out), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv3d_ndhwc_body(x, weight, x_ndhwc, weight_kdrsc, out_ndhwc, out, bias)

            return _conv3d_ndhwc_bias_main

        @T.prim_func
        def _conv3d_ndhwc_main(
            x: T.Tensor((n, c_in, d, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
            x_ndhwc: T.Tensor((n, d, h, w, c_in), dtype),  # type: ignore
            weight_kdrsc: T.Tensor((c_out, kernel_d, kernel_h, kernel_w, c_in), dtype),  # type: ignore
            out_ndhwc: T.Tensor((n, out_d, out_h, out_w, c_out), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            _conv3d_ndhwc_body(x, weight, x_ndhwc, weight_kdrsc, out_ndhwc, out, None)

        return _conv3d_ndhwc_main
