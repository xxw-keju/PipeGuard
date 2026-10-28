# Inductor corpus rerun (branch `rerun`)

The 1164 deduplicated TorchInductor (max-autotune, H100) Triton kernels were
rerun through warmup → `tools/triton_capture.py --grid-ctas 1` →
`pipeguard check --frontend triton --engine maximal`, using the capture and
checker at the head of this branch. The previous column is the 2026-10-27
baseline run.

| file | content |
|---|---|
| `ledger.jsonl` | one row per kernel: id, name, category, models, `outcome`, `previous_outcome`, reason |
| `summary.json` | outcome counts, previous counts, per-category counts |
| `transitions.json` | previous → current outcome counts |
| `cases/<id>/` | every kernel with a hazard verdict: TTGIR, PTX, full `check.json` with witness |
| `fixed/<id>/` | one representative TTGIR per fixed capture gap (PTX where it shows the Membar barriers) |

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

The checker change is a performance fix only: logical-to-other-lane PDG
reachability is answered from synthetic-node descendant/ancestor bitsets
instead of a BFS per query. All 1080 kernels checked by both versions keep
their verdict. The 3 large mma-v2 GEMMs that exceeded the 1200 s check
timeout with the BFS now finish in seconds (SAFE).

## Hazard cases (`cases/`)

| kernels | verdict | witness | status |
|---|---|---|---|
| 27 template GEMMs (wgmma) | SYNC_HAZARD | `WGMMA_RESULT_NOT_READY` (accumulator read before RESULT_READY) | unchanged from the baseline; pending adjudication |
| `15488a949f68ec34` cumsum, `ca3423385c768908` layer_norm_backward | DATA_HAZARD | cross-lane conflict on `in_out_ptr0` / `out_ptr0` | unchanged from the baseline; pending adjudication |
| `a29f20429e2839e6`, `3b81e36797a12305` softmax(+backward) | DATA_HAZARD | `in_out_ptr0` load (`#blocked`) vs store (`#blocked1`) cross-lane overlap | newly reachable via the `ptr<i1>` fix; same family as `15488a94`, suspected capture false positive (store data-depends on the loaded row through `tt.reduce`); pending adjudication |

## Remaining unsupported

50 `MEMORY_ACCESS_UNRESOLVED` (ordinary pointer provenance), 22
`POINTER_PROVENANCE_UNRESOLVED` (negative offset from the external base) and
9 `OPERATION_UNSUPPORTED`, all in pointwise/reduction kernels; every
template GEMM is now checked.
Known open capture item: no WAR barrier is modelled before TMA writes.
