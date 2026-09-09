    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        stride_cm,
    ):
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=[M, K],
            strides=[stride_am, 1],
            block_shape=[32, 128],
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr,
            shape=[K, N],
            strides=[stride_bn, 1],
            block_shape=[32, 128],
        )
        c_desc = tl.make_tensor_descriptor(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, 1],
            block_shape=[32, 32],
        )
        a = a_desc.load([0, 0])
        b = b_desc.load([0, 0]).T
        c = tl.dot(a, b)
        c_desc.store([0, 0], c)
