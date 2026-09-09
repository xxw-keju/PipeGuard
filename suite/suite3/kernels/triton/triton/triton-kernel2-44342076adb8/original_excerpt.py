@pytest.mark.parametrize(None, [None])
@triton.jit
def kernel2(BLOCK_SIZE: tl.constexpr):
    return
