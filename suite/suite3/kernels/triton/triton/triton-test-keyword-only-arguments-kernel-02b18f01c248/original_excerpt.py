    @triton.jit
    def kernel(value=3, *, output, required, optional=5, REQUIRED: tl.constexpr, OPTIONAL: tl.constexpr = 2):
        tl.store(output, _keyword_only_impl(required=required, REQUIRED=REQUIRED))
        tl.store(output + 1,
                 _keyword_only_impl(value, required=required, optional=optional, REQUIRED=REQUIRED, OPTIONAL=OPTIONAL))
        tl.store(output + 2, _keyword_only_varargs(value, required, optional, scale=OPTIONAL))
