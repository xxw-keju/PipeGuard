    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_depthwise_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_depthwise_body(x, weight, out, bias):
            out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)
            with T.Kernel(
                T.ceildiv(out_hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for kh, kw in T.grid(kernel_h, kernel_w):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        spatial_idx = bx * block_n + j
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        valid = (
                            (oc < c_out)
                            & (spatial_idx < out_hw)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (ih < h)
                            & (iw < w)
                        )
                        out_local[i, j] += T.if_then_else(
                            valid,
                            T.cast(x[bz, oc, ih, iw], accum_dtype)
                            * T.cast(weight[oc, 0, kh, kw], accum_dtype),
                            T.cast(0.0, accum_dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if oc < c_out and spatial_idx < out_hw:
                        if has_bias:
                            out_flat[bz, oc, spatial_idx] = T.cast(
                                out_local[i, j] + T.cast(bias[oc], accum_dtype),
                                dtype,
                            )
                        else:
                            out_flat[bz, oc, spatial_idx] = T.cast(out_local[i, j], dtype)

        if has_bias:

            @T.prim_func
            def _conv2d_depthwise_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, 1, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_depthwise_body(x, weight, out, bias)

            return _conv2d_depthwise_bias_main

        @T.prim_func
        def _conv2d_depthwise_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, 1, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ):
            _conv2d_depthwise_body(x, weight, out, None)

        return _conv2d_depthwise_main
