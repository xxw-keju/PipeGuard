    @triton.jit
    def _namedtuple_float_tuple_kernel():
        x, y = float("-inf"), float("inf")  # noqa: F841
