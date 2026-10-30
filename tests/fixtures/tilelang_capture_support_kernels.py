import hashlib
import json
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


# Each lane reads A only where its guard holds: lane 0 never reads A[-1].
@T.prim_func
def guarded_load_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        B[tx] = T.if_then_else(tx >= 1, A[tx - 1], T.float32(0))


# The 64-row box is stored to a 48-row tensor: rows 48..63 are dropped.
@T.prim_func
def descriptor_tma_store_kernel(D: T.Tensor((48, 64), T.float16)):
    with T.Kernel(1, threads=128):
        tile = T.alloc_shared((64, 64), T.float16)
        T.fill(tile, 1.0)
        T.copy(tile, D[0:64, 0:64])


# Rasterized blocks: each CTA writes the 16x16 tile of C it is remapped to.
@T.prim_func
def swizzled_blocks_kernel(C: T.Tensor((64, 64), T.float32)):
    with T.Kernel(4, 4, threads=128) as (bx, by):
        T.use_swizzle(panel_size=2)
        for i, j in T.Parallel(16, 16):
            C[by * 16 + i, bx * 16 + j] = T.float32(1)


# A full-warp register shuffle has no memory footprint.
@T.prim_func
def warp_shuffle_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        value = T.alloc_local((1,), T.float32)
        value[0] = A[tx]
        value[0] = value[0] + T.shfl_xor(value[0], 1)
        B[tx] = value[0]


# Lane 0's predicated cp.async zero-fills S[0] without reading A[-1].
@T.prim_func
def predicated_cp_async_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.float32)
        T.ptx_cp_async(
            T.tvm_access_ptr(T.type_annotation(T.float32), S.data, tx, 1, 2),
            T.tvm_access_ptr(T.type_annotation(T.float32), A.data, tx - 1, 1, 1),
            1,
            predicate=tx >= 1,
        )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.sync_threads()
        B[tx] = S[127 - tx]


# ldmatrix.x4: lane i supplies the address of row i; every lane gets four
# 32-bit fragments.
@T.prim_func
def ldmatrix_kernel(B: T.Tensor((256,), T.bfloat16)):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        S = T.alloc_shared((256,), T.bfloat16)
        fragment = T.alloc_local((8,), T.bfloat16)
        for i in T.serial(8):
            S[tx * 8 + i] = T.bfloat16(1)
        T.sync_threads()
        T.evaluate(tvm.tirx.call_intrin(
            "handle",
            tvm.tirx.op.Op.get("tl.ptx_ldmatrix"),
            False,
            4,
            T.tvm_access_ptr(T.type_annotation(T.bfloat16), S.data, tx * 8, 8, 1),
            T.tvm_access_ptr(T.type_annotation(T.bfloat16), fragment.data, 0, 8, 2),
        ))
        for i in T.serial(8):
            B[tx * 8 + i] = fragment[i]


# TileLang declares float32 ldmatrix operands two registers wider than the
# instruction touches; only the 16-byte row and four registers are accessed.
@T.prim_func
def ldmatrix_wide_kernel(B: T.Tensor((128,), T.float32)):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.float32)
        fragment = T.alloc_local((4,), T.float32)
        for i in T.serial(4):
            S[tx * 4 + i] = T.float32(1)
        T.sync_threads()
        T.evaluate(tvm.tirx.call_intrin(
            "handle",
            tvm.tirx.op.Op.get("tl.ptx_ldmatrix"),
            False,
            4,
            T.tvm_access_ptr(T.type_annotation(T.float32), S.data, tx * 4, 8, 1),
            T.tvm_access_ptr(T.type_annotation(T.float32), fragment.data, 0, 8, 2),
        ))
        for i in T.serial(4):
            B[tx * 4 + i] = fragment[i]


# Threads 32..95 write, meet on named barrier 3 of 64 threads, and read each
# other's elements; threads 0..31 never reach the barrier.
@T.prim_func
def named_barrier_kernel(B: T.Tensor((64,), T.float32)):
    with T.Kernel(1, threads=96):
        tx = T.get_thread_binding()
        S = T.alloc_shared((64,), T.float32)
        if tx >= 32:
            S[tx - 32] = T.float32(1)
            T.evaluate(tvm.tirx.call_intrin(
                "handle", tvm.tirx.op.Op.get("tirx.tvm_storage_sync"), "shared", 3, 64
            ))
            B[tx - 32] = S[95 - tx]


# mma.sync on register fragments after a grid dependency wait.
@T.prim_func
def register_mma_kernel(C: T.Tensor((128,), T.float32)):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        A_local = T.alloc_local((4,), T.float32)
        B_local = T.alloc_local((2,), T.float32)
        C_local = T.alloc_local((4,), T.float32)
        T.evaluate(tvm.tirx.call_intrin("handle", tvm.tirx.op.Op.get("tl.pdl_sync")))
        for i in T.serial(4):
            A_local[i] = T.float32(1)
            C_local[i] = T.float32(0)
        for i in T.serial(2):
            B_local[i] = T.float32(1)
        T.ptx_mma(
            T.float32, "m16n8k8", "row", "col", "tf32", "tf32", "fp32",
            A_local.data, 0, B_local.data, 0, C_local.data, 0, False,
        )
        for i in T.serial(4):
            C[tx * 4 + i] = C_local[i]


# wait_group 8 with one pending group waits for nothing: S is read while the
# copy may still be in flight.
@T.prim_func
def deep_wait_group_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.float32)
        T.ptx_cp_async(
            T.tvm_access_ptr(T.type_annotation(T.float32), S.data, tx, 1, 2),
            T.tvm_access_ptr(T.type_annotation(T.float32), A.data, tx, 1, 1),
            4,
        )
        T.ptx_commit_group()
        T.ptx_wait_group(8)
        T.sync_threads()
        B[tx] = S[127 - tx]



# Loaded scalars bound once and used only as stored values stay opaque.
@T.prim_func
def opaque_scalar_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
    C: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        scale = A[tx] * T.float32(2)
        B[tx] = scale
        C[tx] = scale + T.float32(1)


# A loaded scalar used as an address cannot be resolved.
@T.prim_func
def opaque_address_kernel(
    A: T.Tensor((128,), T.int32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        index = A[tx]
        B[index] = T.float32(1)
        B[127 - index] = T.float32(2)


# Threads 0..127 copy A into S and arrive on the mbarrier once their copies
# land; threads 128..255 wait on it and read each other's elements.
@T.prim_func
def cp_async_noinc_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=256):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.float32)
        bar = T.alloc_buffer((1,), "uint64", scope="shared.barrier")
        if tx == 0:
            T.ptx_init_barrier_thread_count(bar[0], 128)
        T.ptx_fence_barrier_init()
        T.sync_threads()
        if tx < 128:
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.float32), S.data, tx, 1, 2),
                T.tvm_access_ptr(T.type_annotation(T.float32), A.data, tx, 1, 1),
                1,
            )
            T.ptx_commit_group()
            T.evaluate(tvm.tirx.call_intrin(
                "handle", tvm.tirx.op.Op.get("tl.ptx_cp_async_barrier_noinc"), bar[0]
            ))
        else:
            T.mbarrier_wait_parity(bar[0], 0)
            B[tx - 128] = S[255 - tx]


# Each warp's lanes exchange shared elements across __syncwarp.
@T.prim_func
def sync_warp_kernel(
    A: T.Tensor((64,), T.float32),
    B: T.Tensor((64,), T.float32),
):
    with T.Kernel(1, threads=64):
        tx = T.get_thread_binding()
        S = T.alloc_shared((64,), T.float32)
        S[tx] = A[tx]
        T.evaluate(tvm.tirx.call_intrin("handle", tvm.tirx.op.Op.get("tl.sync_warp")))
        B[tx] = S[tx // 32 * 32 + 31 - tx % 32]


# TVM's lowered warp all-reduce: activemask and shuffles are register-only.
@T.prim_func
def warp_allreduce_kernel(
    A: T.Tensor((32,), T.float32),
    B: T.Tensor((32,), T.float32),
):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        red = T.alloc_local((1,), T.float32)
        mask = T.alloc_local((1,), T.uint32)
        with T.attr(
            T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            red[0] = A[tx]
            mask[0] = T.tvm_warp_activemask()
            red[0] = red[0] + T.tvm_warp_shuffle_down(mask[0], red[0], 16, 32, 32)
        B[tx] = red[0]


# Every lane adds into one of four shared counters; the counters return
# their old values to lanes 0..3.
@T.prim_func
def atomic_add_kernel(
    A: T.Tensor((128,), T.float32),
    C: T.Tensor((4,), T.float32),
    D: T.Tensor((4,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        T.atomic_add(C[tx % 4], A[tx])
        T.atomic_max(C[0], A[tx])
        if tx < 4:
            D[tx] = T.atomic_add(C[tx], T.float32(1), return_prev=True)


# Lane tx reads a counter another lane may still be adding into.
@T.prim_func
def atomic_race_kernel(
    A: T.Tensor((128,), T.float32),
    C: T.Tensor((4,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        T.atomic_add(C[tx % 4], A[tx])
        B[tx] = C[(tx + 1) % 4]


# Each lane fast-logs its row of A into registers and adds it to one of four
# shared rows with a single float4 atomic.
@T.prim_func
def vector_atomic_kernel(
    A: T.Tensor((128, 4), T.float32),
    C: T.Tensor((4, 4), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        value = T.alloc_local((4,), T.float32)
        for i in T.serial(4):
            value[i] = T.__log(A[tx, i])
        T.atomic_addx4(C[tx % 4, 0:4], value[0:4])


# A vectorized tile atomic add: the value operand is a shared vector load.
@T.prim_func
def vectorized_atomic_kernel(
    A: T.Tensor((16, 16), T.float32),
    C: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=64):
        A_shared = T.alloc_shared((16, 16), T.float32)
        T.copy(A, A_shared)
        for i, j in T.Parallel(16, 16):
            T.atomic_add(C[i, j], A_shared[i, j])


# A data-dependent branch that only updates the lane's own register.
@T.prim_func
def register_branch_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        best = T.alloc_local((1,), T.float32)
        best[0] = A[tx]
        if best[0] < A[127 - tx]:
            best[0] = A[127 - tx]
        B[tx] = best[0]


# A data-dependent branch that writes shared memory stays unsupported.
@T.prim_func
def shared_branch_kernel(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        tx = T.get_thread_binding()
        S = T.alloc_shared((128,), T.float32)
        S[tx] = T.float32(0)
        if A[tx] > T.float32(0):
            S[tx] = A[tx]
        T.sync_threads()
        B[tx] = S[127 - tx]


# Warp votes and matches are register-only exchanges.
@T.prim_func
def warp_vote_kernel(
    A: T.Tensor((32,), T.float32),
    B: T.Tensor((32,), T.uint32),
):
    with T.Kernel(1, threads=32):
        tx = T.get_thread_binding()
        positive = A[tx] > T.float32(0)
        B[tx] = (
            T.Cast("uint32", T.ballot_sync(0xFFFFFFFF, positive))
            + T.Cast("uint32", T.any_sync(0xFFFFFFFF, positive))
            + T.match_any_sync(0xFFFFFFFF, tx)
        )


target = tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
})
for function, name in (
    (guarded_load_kernel, "tilelang_guarded_load_device.tvm.json"),
    (swizzled_blocks_kernel, "tilelang_swizzled_blocks_device.tvm.json"),
    (warp_shuffle_kernel, "tilelang_warp_shuffle_device.tvm.json"),
    (predicated_cp_async_kernel, "tilelang_predicated_cp_async_device.tvm.json"),
    (ldmatrix_kernel, "tilelang_ldmatrix_device.tvm.json"),
    (ldmatrix_wide_kernel, "tilelang_ldmatrix_wide_device.tvm.json"),
    (named_barrier_kernel, "tilelang_named_barrier_device.tvm.json"),
    (register_mma_kernel, "tilelang_register_mma_device.tvm.json"),
    (deep_wait_group_kernel, "tilelang_deep_wait_group_device.tvm.json"),
    (opaque_scalar_kernel, "tilelang_opaque_scalar_device.tvm.json"),
    (opaque_address_kernel, "tilelang_opaque_address_device.tvm.json"),
    (cp_async_noinc_kernel, "tilelang_cp_async_noinc_device.tvm.json"),
    (sync_warp_kernel, "tilelang_sync_warp_device.tvm.json"),
    (warp_allreduce_kernel, "tilelang_warp_allreduce_device.tvm.json"),
    (atomic_add_kernel, "tilelang_atomic_add_device.tvm.json"),
    (atomic_race_kernel, "tilelang_atomic_race_device.tvm.json"),
    (vector_atomic_kernel, "tilelang_vector_atomic_device.tvm.json"),
    (vectorized_atomic_kernel, "tilelang_vectorized_atomic_device.tvm.json"),
    (register_branch_kernel, "tilelang_register_branch_device.tvm.json"),
    (shared_branch_kernel, "tilelang_shared_branch_device.tvm.json"),
    (warp_vote_kernel, "tilelang_warp_vote_device.tvm.json"),
    (descriptor_tma_store_kernel, "tilelang_descriptor_tma_store_device.tvm.json"),
):
    with target:
        artifact = tilelang.lower(
            function,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    text = tvm.ir.save_json(artifact.device_mod)
    Path(name).write_text(text)
Path("tilelang_descriptor_tma_store_descriptors.json").write_text(json.dumps({
    "evidence_version": "tilelang-tma-descriptor-evidence-v1",
    "source_ir_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    "descriptors": {
        "D_desc": {
            "allocation_id": "global:D",
            "element_bytes": 2,
            "shape": [48, 64],
            "strides": [64, 1],
            "tile_shape": [64, 64],
            "coordinate_order": [1, 0],
        },
    },
}, sort_keys=True, separators=(",", ":")))
