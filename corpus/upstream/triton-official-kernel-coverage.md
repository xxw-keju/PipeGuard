# Triton official useful-kernel coverage at `61c02bcd`

This document defines and records the bounded PipeGuard coverage claim for Triton revision
`61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f`.

## Denominator

A **useful kernel root** is a device function directly launched by the maintained
`python/triton_kernels/triton_kernels` package or by a standard Triton-language tutorial under
`python/tutorials/`. Functions that are only called from another JIT function are helpers and are
not counted as roots. Test-only kernels under `python/test/` and `python/triton_kernels/tests/` are
not production/tutorial roots. Gluon examples are a different frontend and are not counted as
Triton-language roots.

This definition yields 51 roots:

- 28 product roots (the 29 bracket-launched definitions in `triton_kernels`, minus the
  `_random_block_signs` testing utility);
- 23 standard Triton tutorial roots.

Of these, 28 roots have at least one specialization in the static NVIDIA SM90 PipeGuard profile.
The other 23 roots have no specialization in that profile for the reasons listed below. A root may
also have both checked specializations and excluded dynamic/atomic specializations.

The checked profile requires a finite statically selected CTA path, exact pointer/descriptor alias
and address provenance, supported SM90 operations, and no runtime-loaded value that determines a
memory address or loop/control bound. The real launch grid is retained and CTA `(0,0,0)` is selected
unless a branch-specific check explicitly records another CTA. Atomic RMW, multi-GPU remote-memory
protocols, multi-kernel PDL, AMD, Blackwell/CLC, and Gluon are outside this claim.

## Checked root implementations

### `triton_kernels` product package: 12/12 profile-included roots

| Root | Covered specialization classes |
| --- | --- |
| `_matmul` | regular/batched; BF16, FP16, MXFP4, MXFP8; 4/8 warps; block-M 16/128 |
| `_p_matmul` | persistent TMA; the same static dtype/warp/block classes; boundary OOB zero-fill/discard |
| `_downcast_to_mxfp` | BF16 to MXFP4 and MXFP8 |
| `_upcast_from_mxfp` | MXFP4 and MXFP8 to BF16 with tensor descriptors |
| `_reduce_forward` | dimensions 0/1/2, no-mask and scale branches; static noinline callee expansion |
| `_reduce_backward` | no-mask branch |
| `_swiglu` | regular BF16 non-routing path, compiler-generated cp.async pipeline |
| `_topk_forward` | streaming TopK with and without softmax and two block-size classes |
| `_bitmatrix_metadata_compute_stage1` | partial-sum, column-offset, and memset CTA roles |
| `_convert_bits_kernel` | forward and inverse Hopper FP4 bit conversion |
| `_swizzle_from_strided_kernel` | MMA v2/v3 and both packing axes |
| `_unswizzle_to_strided_kernel` | MMA v2/v3 and both packing axes |

### Standard Triton tutorials: 16/16 profile-included roots

| Tutorial roots | Covered specialization classes |
| --- | --- |
| `01-vector-add.add_kernel` | FP32 vector add |
| `02-fused-softmax.softmax_kernel` | FP32 row softmax with compiler-generated cp.async |
| `03-matrix-multiplication.matmul_kernel` | FP16 WGMMA/cp.async matmul |
| `04-low-memory-dropout._dropout`, `_seeded_dropout` | explicit-mask and Philox-seeded dropout |
| `05-layer-norm._layer_norm_fwd_fused`, `_layer_norm_bwd_dwdb` | forward and final weight/bias reduction |
| `06-fused-attention._attn_fwd`, `_attn_bwd_preprocess`, `_attn_bwd` | causal/noncausal, head 64/128, context 128/1024 |
| `07-extern-functions.asin_kernel` | pure libdevice elementwise operation plus global load/store |
| `09-persistent-matmul.matmul_kernel` | pointer cp.async/WGMMA path |
| `09-persistent-matmul.matmul_kernel_tma` | descriptor TMA path |
| `09-persistent-matmul.matmul_kernel_persistent` | persistent pointer path |
| `09-persistent-matmul.matmul_kernel_tma_persistent` | TMA persistent, warp-specialize false/true |
| `09-persistent-matmul.matmul_kernel_descriptor_persistent` | compiler-created descriptors and warp specialization |

The accepted evidence set contains 95 complete maximal-checker executions and 84 unique TTGIR
modules after SHA-256 deduplication. All 95 return `SAFE`; together they contain 1,213,276 emitted
memory/protocol events. The count includes the production matmul/attention matrix and the
root-completion specializations above, but excludes failed probes and all profile-external cases.
The aggregate record is `/tmp/pipeguard-search/official-useful-coverage-records.json` (SHA-256
`a4db09aa8a7f7a92dd575382a9220a63ee5db2184e0af2ed83645cae5928dde7`).

No current-revision concurrency bug was found in this profile. The accepted specialization probes
use compile-warmup/native lowering only; they do not launch a GPU kernel. Generated TTGIR,
evidence, snapshots, PGIR, and checker output stay under `/tmp/pipeguard-search/` and are not
committed.

## Roots wholly outside the checked profile

### Product package: 16 roots

| Roots | Count | Exclusion evidence |
| --- | ---: | --- |
| `_masked_compaction`, `_create_row_idxs`, `_topk_backward`, `_bitmatrix_metadata_compute_stage2`, `_ragged_tensor_metadata_memset`, `_ragged_tensor_metadata_compute`, `_remap_ragged_tensor_metadata` | 7 | Runtime-loaded masks, indices, slice sizes, or schedules determine addresses or loop/control bounds. |
| `_convert_dp_to_ep`, `_convert_ep_to_dp` | 2 | Runtime rank/expert selection and peer pointers; multi-GPU remote-memory protocol is outside the single-device profile. |
| `_fpsan_kernel` | 1 | Gluon frontend rather than the Triton frontend. |
| `_sum_bitmatrix_rows` | 1 | Cross-CTA `atomic_add`; atomic RMW is unsupported. |
| `_swizzle_blackwell_mx_scale`, `pad_segments_kernel`, `unpad_segments_kernel`, `_convert_shuffled_mxfp4` | 4 | Blackwell-specific physical formats/profile. |
| `_matmul_flops_and_bytes_from_slices_kernel` | 1 | Internal profiling-metadata helper, excluded from the useful-kernel denominator. |

### Standard tutorials: 7 roots

| Roots | Count | Exclusion evidence |
| --- | ---: | --- |
| `08-grouped-gemm.grouped_matmul_kernel`, `grouped_matmul_tma_kernel` | 2 | Runtime group metadata determines matrix pointers, dimensions, and loop bounds. |
| `05-layer-norm._layer_norm_bwd_dx_fused` | 1 | Uses `tt.atomic_cas`/atomic accumulation. |
| `09-persistent-matmul.matmul_kernel_tma_clc` | 1 | Blackwell-only CLC scheduling. |
| `10-block-scaled-matmul.block_scaled_matmul_kernel` | 1 | NVIDIA SM100+ block-scaled MMA path. |
| `10-block-scaled-matmul.block_scaled_matmul_kernel_cdna4` | 1 | AMD CDNA4 path. |
| `11-programmatic-dependent-launch.add_kernel` | 1 | Grid-dependency control is an inter-kernel protocol outside single-kernel PGIR. |

## Excluded specialization families of otherwise checked roots

- `_matmul`/`_p_matmul`: ragged slice metadata, gather/scatter indices, remote peer pointers, flex
  scale accumulation, and atomic checksum variants.
- `_reduce_forward`/`_reduce_backward`: mask-dependent accesses, row-index scheduling driven by
  loaded masks, device-loaded unpadded bounds, and flex/atomic scale variants.
- `_swiglu`: routing path whose device `NTokens` value changes the persistent loop bound, and flex
  output-scale atomic updates.
- `_topk_forward`: provided-index gathering and all-gather peer-memory variants.

These are exclusions from the static/no-atomic profile, not failed attempts counted as `SAFE`.

## Frontend gaps closed by this sweep

The production and tutorial sweep added exact static support for pointer tensor broadcast, selected
`scf.if`, finite `cf.cond_br`, loop-carried async and register provenance, empty WGMMA waits,
SM90 cp.async capacity 16, contiguous cp.async coalescing, TMA boundary zero-fill/discard, pure
`tt.bitcast`, `ub.poison`, `tt.fp_to_fp`, `tt.clampf`, `ttg.fp4_to_fp`, `tt.extern_elementwise`, and
`tt.mulhiui`, plus finite expansion of zero-result private `tt.call` helpers. Unknown aliases,
dynamic control/addresses, recursive or value-returning calls, and unsupported effects still fail
closed.
