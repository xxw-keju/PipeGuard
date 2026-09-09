@triton.jit
def kernel_no_arg_print():
    print("", tl.program_id(0))
