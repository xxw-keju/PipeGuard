# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import NamedTuple, Optional
import tilelang
import tilelang.language as T
_NUM_ACCUMULATORS = 4
_KEY_MAGNITUDE = 2147483647
_KEY_NAN = 2147483647
_KEY_IDENTITY = -2 ** 31

def _ordering_key(op_kind: str):
    """Build the macro turning one element into the int32 key the reduction ranks by.

    Ranking bit patterns collapses PyTorch's ordering -- NaN over every number, two NaNs
    equal, ties to the lower index -- into one integer compare. Spelled out per element
    it costs a dozen instructions, and this kernel is short of its bandwidth by about
    the same factor. Callers bind the result to a local before ranking it: a macro
    argument is substituted at every mention, so passing the call directly would
    recompute the key once per comparison the merge makes.

    Negating first turns argmin into argmax and leaves a NaN a NaN, so NaN outranks in
    both. The key is the magnitude bits, negated in two's complement when the sign bit
    is set: that orders the whole line, and it sends both zeros to 0 without a test,
    which is what lets the index break their tie the way PyTorch does. A positive NaN's
    magnitude already outranks +inf's; the negated ones do not, so NaN is answered
    outright.
    """
    negate = op_kind == 'argmin'

    @T.macro
    def key_of(value):
        cast = T.cast(value, 'float32')
        oriented = -cast if negate else cast
        bits = T.reinterpret(oriented, 'int32')
        sign = bits >> 31
        magnitude = T.bitwise_and(bits, T.int32(_KEY_MAGNITUDE))
        ordered = T.bitwise_xor(magnitude, sign) - sign
        return T.if_then_else(oriented != oriented, T.int32(_KEY_NAN), ordered)
    return key_of

class _PairOps(NamedTuple):
    set_identity: object
    init_accumulators: object
    update: object
    advance: object
    merge_accumulators: object
    warp_reduce: object

def _make_pair_ops(op_kind: str, n: int):
    """Create argreduce-local pair operations shared by all launch paths."""

    @T.macro
    def set_identity(keys, indices, slot):
        keys[slot] = T.int32(_KEY_IDENTITY)
        indices[slot] = T.int32(n)

    @T.macro
    def init_accumulators(keys, indices):
        for accumulator in T.serial(_NUM_ACCUMULATORS):
            set_identity(keys, indices, accumulator)

    @T.macro
    def update(keys, indices, slot, candidate_key, candidate_index):
        """Merge a candidate into a slot: higher key wins, an equal key breaks low."""
        if candidate_key > keys[slot] or (candidate_key == keys[slot] and candidate_index < indices[slot]):
            keys[slot] = candidate_key
            indices[slot] = T.cast(candidate_index, 'int32')

    @T.macro
    def advance(keys, indices, slot, candidate_key, candidate_index):
        """Merge a candidate reached after everything in the slot: one key comparison.

        A streaming loop hands a slot ascending indices, so an equal key cannot win.
        """
        if candidate_key > keys[slot]:
            keys[slot] = candidate_key
            indices[slot] = T.cast(candidate_index, 'int32')

    @T.macro
    def merge_accumulators(keys, indices, best_key, best_index):
        best_key[0] = keys[0]
        best_index[0] = indices[0]
        for accumulator in T.serial(1, _NUM_ACCUMULATORS):
            update(best_key, best_index, 0, keys[accumulator], indices[accumulator])

    @T.macro
    def warp_reduce(best_key, best_index, stages, width):
        for stage in T.serial(stages):
            mask = T.int32(width // 2) >> stage
            update(best_key, best_index, 0, T.shfl_xor(best_key[0], mask, width=width), T.shfl_xor(best_index[0], mask, width=width))
    return _PairOps(set_identity=set_identity, init_accumulators=init_accumulators, update=update, advance=advance, merge_accumulators=merge_accumulators, warp_reduce=warp_reduce)

def _resolve_kernel(**pipeguard_bindings):
    M = pipeguard_bindings['M']
    N = pipeguard_bindings['N']
    dtype = pipeguard_bindings['dtype']
    items_per_iteration = pipeguard_bindings['items_per_iteration']
    iterations = pipeguard_bindings['iterations']
    lanes = pipeguard_bindings['lanes']
    log_lanes = pipeguard_bindings['log_lanes']
    op_kind = pipeguard_bindings['op_kind']

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], out: T.Tensor[(M,), 'int64']):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * block_m + tx // lanes
                lane = tx % lanes
                keys = T.alloc_local((_NUM_ACCUMULATORS,), 'int32')
                indices = T.alloc_local((_NUM_ACCUMULATORS,), 'int32')
                candidate = T.alloc_local((1,), 'int32')
                best_key = T.alloc_local((1,), 'int32')
                best_index = T.alloc_local((1,), 'int32')
                ops.init_accumulators(keys, indices)
                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = iteration * items_per_iteration + accumulator * lanes + lane
                        if row < M and index < N:
                            candidate[0] = key_of(x[row, index])
                            ops.advance(keys, indices, accumulator, candidate[0], index)
                ops.merge_accumulators(keys, indices, best_key, best_index)
                ops.warp_reduce(best_key, best_index, log_lanes, lanes)
                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], 'int64')
        return main
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    factory = _resolve_kernel(**case["build"]["factory_arguments"])
    return factory(**case["build"]["kernel_arguments"])
