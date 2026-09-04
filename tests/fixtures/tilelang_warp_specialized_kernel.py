import hashlib
import json
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    num_stages = 2
    mbarrier_list = [128, 128] * num_stages
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        mbars = T.alloc_barrier(mbarrier_list)
        with T.ws(0):
            T.clear(C_local)
        for ko in range(T.ceildiv(K, block_K)):
            with T.ws(1):
                T.mbarrier_wait_parity(
                    mbarrier=mbars[ko % num_stages + num_stages],
                    parity=((ko // num_stages) % num_stages) ^ 1,
                )
                T.tma_copy(
                    A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                    A_shared[ko % num_stages, :, :],
                    barrier=mbars[ko % num_stages],
                )
                T.tma_copy(
                    B[ko * block_K:(ko + 1) * block_K, bx * block_N:(bx + 1) * block_N],
                    B_shared[ko % num_stages, :, :],
                    barrier=mbars[ko % num_stages],
                )
                T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
            with T.ws(0):
                T.mbarrier_wait_parity(
                    mbarrier=mbars[ko % num_stages],
                    parity=(ko // num_stages) % num_stages,
                )
                T.gemm(
                    A_shared[ko % num_stages, :, :],
                    B_shared[ko % num_stages, :, :],
                    C_local,
                )
                T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])
        with T.ws(0):
            T.copy(C_local, C[by * block_M, bx * block_N])
    return C


function = matmul.get_tir(
    M=64,
    N=16,
    K=16,
    block_M=64,
    block_N=16,
    block_K=16,
)
target = tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
})
with target:
    artifact = tilelang.lower(
        function,
        target=target,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
text = tvm.ir.save_json(artifact.device_mod)
Path("tilelang_warp_specialized_device.tvm.json").write_text(text)
Path("tilelang_warp_specialized_descriptors.json").write_text(json.dumps({
    "evidence_version": "tilelang-tma-descriptor-evidence-v1",
    "source_ir_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    "descriptors": {
        "A_desc": {
            "allocation_id": "global:A",
            "element_bytes": 2,
            "shape": [64, 16],
            "strides": [16, 1],
            "tile_shape": [64, 16],
            "coordinate_order": [1, 0],
        },
        "B_desc": {
            "allocation_id": "global:B",
            "element_bytes": 2,
            "shape": [16, 16],
            "strides": [16, 1],
            "tile_shape": [16, 16],
            "coordinate_order": [1, 0],
        },
    },
}, sort_keys=True, separators=(",", ":")))
