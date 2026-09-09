# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import math
import tilelang

class GroupTiling:
    """Tiles enumerated per group, for ``num_groups`` groups of ``block_m`` rows.

    Example:
        ```python linenums="1"
        tiling = GroupTiling(num_groups=16, block_m=64)
        tiling.tile_upper_bound(4096) # 80
        ```
    """

    def __init__(self, num_groups: int, block_m: int) -> None:
        self.num_groups = num_groups
        self.block_m = block_m
        self._search_steps = max(1, (num_groups - 1).bit_length())

    def tile_upper_bound(self, numel: int) -> int:
        """Tiles ``numel`` rows can need, at most one partial tile per group.

        Sizes arrive on the device, so a grid takes this bound and drops the
        tiles past the count the kernel finds at run time.
        """
        return numel // self.block_m + self.num_groups

    @property
    def cumsum(self):
        """Macro writing the per-group tile-count prefix sum into ``s_cum``.

        ``s_cum[g + 1] - s_cum[g]`` is the tiles group *g* needs, and
        ``s_cum[num_groups]`` is how many the call has.
        """
        num_groups, block_m = (self.num_groups, self.block_m)

        @T.macro
        def group_tile_cumsum(sizes, s_cum):
            s_cum[0] = T.int32(0)
            for g in T.serial(num_groups):
                s_cum[g + 1] = s_cum[g] + (sizes[g] + T.int32(block_m - 1)) // T.int32(block_m)
        return group_tile_cumsum

    @property
    def decode(self):
        """Macro searching ``s_cum`` for the group owning one M tile id.

        ``row`` is the tile's first row within its group: the caller reads the
        packed start as ``offsets[group] + row``.
        """
        num_groups, block_m, steps = (self.num_groups, self.block_m, self._search_steps)

        @T.macro
        def group_tile_decode(m_tile, s_cum, lo, hi, group, row):
            lo[0] = T.int32(0)
            hi[0] = T.int32(num_groups - 1)
            for _ in T.serial(steps):
                mid = lo[0] + hi[0] >> T.int32(1)
                if s_cum[mid + 1] <= m_tile:
                    lo[0] = mid + T.int32(1)
                else:
                    hi[0] = mid
            group[0] = lo[0]
            row[0] = (m_tile - s_cum[lo[0]]) * T.int32(block_m)
        return group_tile_decode

def _kernel_factory(batch_sum, batch_count, N, K, transpose_a, transpose_b, dtype='float16'):
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _grouped_gemm_func(block_m, block_n, block_k, num_stages, threads):
        if not transpose_a:
            A_shape = (batch_sum, K)
            C_shape = (batch_sum, N)
            A_shared_shape = (block_m, block_k)
            if transpose_b:
                B_shape = (batch_count, N, K)
                B_shared_shape = (block_n, block_k)
            else:
                B_shape = (batch_count, K, N)
                B_shared_shape = (block_k, block_n)
            _tiling = GroupTiling(batch_count, block_m)
            _num_pid_m = _tiling.tile_upper_bound(batch_sum)
            _num_pid_n = math.ceil(N / block_n)

            @T.prim_func
            def _grouped_gemm_main(A: T.Tensor(A_shape, dtype), B: T.Tensor(B_shape, dtype), C: T.Tensor(C_shape, dtype), batch_sizes: T.Tensor([batch_count], 'int32'), batch_offsets: T.Tensor([batch_count], 'int32'), batch_padded_offsets: T.Tensor([batch_count], 'int32')):
                with T.Kernel(_num_pid_m * _num_pid_n, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    s_cum = T.alloc_shared([batch_count + 1], 'int32')
                    lo = T.alloc_local([1], 'int32')
                    hi = T.alloc_local([1], 'int32')
                    row = T.alloc_local([1], 'int32')
                    cur_batch_idx = T.alloc_local([1], 'int32')
                    _tiling.cumsum(batch_sizes, s_cum)
                    m_tile = pid // _num_pid_n
                    n_start = pid % _num_pid_n * block_n
                    if m_tile < s_cum[batch_count]:
                        _tiling.decode(m_tile, s_cum, lo, hi, cur_batch_idx, row)
                        m_start = batch_offsets[cur_batch_idx[0]] + row[0]
                        actual_rows = T.min(block_m, batch_sizes[cur_batch_idx[0]] - row[0])
                        actual_cols = T.min(block_n, N - n_start)
                        T.clear(C_local)
                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(i < actual_rows and j < K - k * block_k, A[m_start + i, k * block_k + j], 0)
                            if transpose_b:
                                for i, j in T.Parallel(block_n, block_k):
                                    B_shared[i, j] = T.if_then_else(j < K - k * block_k and i < actual_cols, B[cur_batch_idx[0], n_start + i, k * block_k + j], 0)
                            else:
                                for i, j in T.Parallel(block_k, block_n):
                                    B_shared[i, j] = T.if_then_else(i < K - k * block_k and j < actual_cols, B[cur_batch_idx[0], k * block_k + i, n_start + j], 0)
                            T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)
                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]
        else:
            A_shape = (batch_sum, N)
            C_shape = (batch_count, N, K)
            A_shared_shape = (block_m, block_n)
            if transpose_b:
                B_shape = (K, batch_sum)
                B_shared_shape = (block_k, block_m)
            else:
                B_shape = (batch_sum, K)
                B_shared_shape = (block_m, block_k)

            @T.prim_func
            def _grouped_gemm_main(A: T.Tensor(A_shape, dtype), B: T.Tensor(B_shape, dtype), C: T.Tensor(C_shape, dtype), batch_sizes: T.Tensor([batch_count], 'int32'), batch_offsets: T.Tensor([batch_count], 'int32'), batch_padded_offsets: T.Tensor([batch_count], 'int32')):
                with T.Kernel(batch_count, T.ceildiv(N, block_n) * T.ceildiv(K, block_k), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_n, block_k], accum_dtype)
                    n_block_idx = by // T.ceildiv(K, block_k)
                    k_block_idx = by % T.ceildiv(K, block_k)
                    n_start = n_block_idx * block_n
                    k_start = k_block_idx * block_k
                    actual_N = T.min(block_n, N - n_start)
                    actual_K = T.min(block_k, K - k_start)
                    T.clear(C_local)
                    batch_start = batch_offsets[bx]
                    batch_size = batch_sizes[bx]
                    for m in T.Pipelined(T.ceildiv(batch_size, block_m), num_stages=num_stages):
                        m_start = batch_start + m * block_m
                        actual_rows = T.min(block_m, batch_size - m * block_m)
                        for i, j in T.Parallel(block_m, block_n):
                            A_shared[i, j] = T.if_then_else(i < actual_rows and j < actual_N, A[m_start + i, n_start + j], 0)
                        if transpose_b:
                            for i, j in T.Parallel(block_k, block_m):
                                B_shared[i, j] = T.if_then_else(i < actual_K and j < actual_rows, B[k_start + i, m_start + j], 0)
                        else:
                            for i, j in T.Parallel(block_m, block_k):
                                B_shared[i, j] = T.if_then_else(i < actual_rows and j < actual_K, B[m_start + i, k_start + j], 0)
                        T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=transpose_b)
                    for i, j in T.Parallel(block_n, block_k):
                        if i < actual_N and j < actual_K:
                            C[bx, n_start + i, k_start + j] = C_local[i, j]
        return _grouped_gemm_main
    return _grouped_gemm_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
