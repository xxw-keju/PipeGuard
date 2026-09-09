@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel3(BLOCK_SIZE: tl.constexpr):
    return
