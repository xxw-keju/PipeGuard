# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang
import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    _tiling = pipeguard_bindings['_tiling']
    max_tiles = pipeguard_bindings['max_tiles']
    num_experts = pipeguard_bindings['num_experts']

    @tilelang.jit(out_idx=[1, 2, 3])
    def _func(threads: int):

        @T.prim_func
        def _sched_main(true_sizes: T.Tensor([num_experts], 'int32'), tile_expert_ids: T.Tensor([max_tiles], 'int32'), tile_row_offsets: T.Tensor([max_tiles], 'int32'), total_tiles: T.Tensor([1], 'int32')):
            with T.Kernel(T.ceildiv(max_tiles, threads), threads=threads) as (bx,):
                s_cum = T.alloc_shared([num_experts + 1], 'int32')
                lo = T.alloc_local([1], 'int32')
                hi = T.alloc_local([1], 'int32')
                _ex = T.alloc_local([1], 'int32')
                _row = T.alloc_local([1], 'int32')
                tx = T.get_thread_binding()
                if tx == 0:
                    _tiling.cumsum(true_sizes, s_cum)
                    if bx == 0:
                        total_tiles[0] = s_cum[num_experts]
                T.sync_threads()
                tile_id = bx * threads + tx
                if tile_id < max_tiles:
                    if tile_id < s_cum[num_experts]:
                        _tiling.decode(tile_id, s_cum, lo, hi, _ex, _row)
                        tile_expert_ids[tile_id] = _ex[0]
                        tile_row_offsets[tile_id] = _row[0]
                    else:
                        tile_expert_ids[tile_id] = T.int32(-1)
                        tile_row_offsets[tile_id] = T.int32(0)
        return _sched_main
    return _func

class _GroupTiling:
    def __init__(self, num_groups, block_m):
        self.num_groups=num_groups; self.block_m=block_m; self._search_steps=max(1,(num_groups-1).bit_length())
    @property
    def cumsum(self):
        num_groups,block_m=self.num_groups,self.block_m
        @T.macro
        def group_tile_cumsum(sizes,s_cum):
            s_cum[0]=T.int32(0)
            for g in T.serial(num_groups): s_cum[g+1]=s_cum[g]+(sizes[g]+T.int32(block_m-1))//T.int32(block_m)
        return group_tile_cumsum
    @property
    def decode(self):
        num_groups,block_m,steps=self.num_groups,self.block_m,self._search_steps
        @T.macro
        def group_tile_decode(m_tile,s_cum,lo,hi,group,row):
            lo[0]=T.int32(0); hi[0]=T.int32(num_groups-1)
            for _ in T.serial(steps):
                mid=(lo[0]+hi[0])>>T.int32(1)
                if s_cum[mid+1]<=m_tile: lo[0]=mid+T.int32(1)
                else: hi[0]=mid
            group[0]=lo[0]; row[0]=(m_tile-s_cum[lo[0]])*T.int32(block_m)
        return group_tile_decode

def build_kernel(case):
    a=dict(case["build"]["factory_arguments"]); a["_tiling"]=_GroupTiling(a["num_experts"],a.pop("block_m"))
    return _resolve_kernel(**a)(**case["build"]["kernel_arguments"])
