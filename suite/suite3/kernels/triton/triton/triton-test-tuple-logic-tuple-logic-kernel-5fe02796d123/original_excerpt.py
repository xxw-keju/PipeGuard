    @triton.jit
    def tuple_logic_kernel():

        # arity-2 BoolOps:
        tl.static_assert(((3, 4) or (5, 6)) == (3, 4))
        tl.static_assert(((3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((3, 4) and ()) == ())
        tl.static_assert((() or (5, 6)) == (5, 6))

        # arity-3 BoolOps:
        tl.static_assert(((1, 2) and (3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((1, 2) or (3, 4) or (5, 6)) == (1, 2))

        # constexpr short-circuiting over dynamic argument:
        tl.static_assert((() and tl.program_id(0)) == ())
