@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return
