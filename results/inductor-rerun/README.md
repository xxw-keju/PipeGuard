# Inductor corpus rerun (branch `rerun`)

The 1164 deduplicated TorchInductor (max-autotune, H100) Triton kernels were
rerun through warmup → `tools/triton_capture.py --grid-ctas 1` →
`pipeguard check --frontend triton --engine maximal`, using the capture and
checker at the head of this branch (kernels whose snapshot changed after the
last capture fixes were recaptured from the saved TTGIR and binding evidence
and rechecked). The previous column is the 2026-10-27 baseline run.

| file | content |
|---|---|
| `ledger.jsonl` | one row per kernel: id, name, category, models, `outcome`, `previous_outcome`, reason |
| `summary.json` | outcome counts, previous counts, per-category counts |
| `transitions.json` | previous → current outcome counts |
| `cases/<id>/` | every kernel with a hazard verdict: TTGIR, PTX, full `check.json` with witness |
| `fixed/<id>/` | one representative TTGIR per fixed capture gap (PTX where it shows the barriers) |
| `adjudication/` | GPU reproduction script and log for the confirmed hazard |

## What changed

All 322 `CP_ASYNC_TRANSFER_WIDTH_UNSUPPORTED` template GEMMs, the 36
`LAYOUT_OWNERSHIP_UNRESOLVED` mma-v2 GEMMs and 59 of the 109
`MEMORY_ACCESS_UNRESOLVED` kernels are now checked. Each `fixed/` entry exercises
one capture change:

| representative | gap | capture change |
|---|---|---|
| `03169881145c7cab` (wgmma, row-major shared) | 2-byte per-element cp.async | per-thread contiguous runs split into aligned 16/8/4-byte transfers (d7e2c264) |
| `004d09614e66ab19` (wgmma, `nvmma_shared transposed=true`) | runs not contiguous in row-major slots | shared memdescs linearized by encoding order (nvmma transposed, swizzled `order`) for cp.async and local_load/local_store |
| `02bf0c8b977d474a` (persistent reduction, `ptr<i1>` mask) | sub-byte pointer provenance | `ptr<i1>` tensors are byte-wide; accesses go through Triton's i8 bitcast, direct sub-byte access still fails closed |
| `04842feff24c5ce3` (mma v2, pipelined cp.async) | dot-operand local_load layout | ldmatrix dot-operand loads modelled as every lane reading the whole tile; Membar barrier after every `async_wait` and WAR barrier before a cp.async reissue (PTX: `cp.async.wait_group; bar.sync` and `bar.sync` before the next stage's copies) |
| `dfe15f4ebb50e226`, `0df94ff12b780f2b` (mma v2, initialized `local_alloc` in loop) | loop-carried WAR on the staged A tile | WAR barrier before an initialized `local_alloc` (PTX: `bar.sync` before the loop back edge) |
| `0a9c88a31289a46f` (wgmma, single K step) | `warp_group_dot` without `isAsync` reported as `WGMMA_RESULT_NOT_READY` | a synchronous `warp_group_dot` lowers to `commit_group; wait_group 0` (PTX), now captured as RESULT_READY/OPERANDS_RELEASED waits |
| `a29f20429e2839e6` (softmax, `in_out_ptr0`) | load/store of the same row in different layouts reported as WAR | a `convert_layout` that moves elements across warps lowers through shared memory with `bar.sync` (PTX), now captured as a CTA barrier; Membar barriers also order generic global accesses, as `bar.sync` does |

The checker change is a performance fix only: logical-to-other-lane PDG
reachability is answered from synthetic-node descendant/ancestor bitsets
instead of a BFS per query. All 1080 kernels checked by both versions keep
their verdict. The 3 large mma-v2 GEMMs that exceeded the 1200 s check
timeout with the BFS now finish in seconds (SAFE).

The last two capture changes alter 170 snapshots; rechecking them moves 27
SYNC_HAZARD and 2 DATA_HAZARD verdicts to SAFE and leaves the other 141
verdicts unchanged (140 SAFE, 1 DATA_HAZARD).

## Hazard cases (`cases/`) and adjudication

| kernels | verdict | adjudication |
|---|---|---|
| `ca3423385c768908` layer_norm_backward | DATA_HAZARD | **real, wrong results.** The first reduction loop stores `out_ptr0` (`#blocked1`), the second loop reloads the same elements (`#blocked`); no barrier between them in the PTX, and row 4 is written by warp 0 and read by warp 2. On H100 with the corpus Triton (3.8) every one of 300 runs gives wrong `out_ptr3` when `out_ptr0` starts as NaN; adding `tl.debug_barrier()` between the loops gives 0/300. Triton 3.2 happens to place `convert_layout`/`reduce` barriers between the store and the reload, so it does not reproduce there (`adjudication/`). |
| `15488a949f68ec34` cumsum | DATA_HAZARD | **real race, benign.** The 1x8 tensor is replicated 16 times; all 128 threads load `in_out_ptr0[tid & 7]`, only threads 0-7 store, and threads 8-127 are unordered with those stores. The replicas' scan results are never stored, so the output is unaffected. |
| 27 template GEMMs (`WGMMA_RESULT_NOT_READY`) | SAFE now | capture false positive (synchronous `warp_group_dot`), fixed above |
| `a29f20429e2839e6`, `3b81e36797a12305` softmax(+backward) | SAFE now | capture false positive (cross-warp `convert_layout` barrier), fixed above |

## Remaining unsupported

50 `MEMORY_ACCESS_UNRESOLVED` (ordinary pointer provenance), 22
`POINTER_PROVENANCE_UNRESOLVED` (negative offset from the external base) and
9 `OPERATION_UNSUPPORTED`, all in pointwise/reduction kernels; every
template GEMM is now checked.
Known open capture item: no WAR barrier is modelled before TMA writes.
