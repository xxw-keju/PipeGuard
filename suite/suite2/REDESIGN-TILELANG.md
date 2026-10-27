# Suite 2 TileLang structural-diversity redesign (v2)


Redesign of the TileLang half of the Suite 2 labeled corpus to eliminate the
structural duplication found by `tools/suite2_structural_audit.py`
(pure shape-duplicate cells and repeat-count ladders). Every
`(tilelang, primitive, result_class)` cell now holds five structurally
distinct synchronization-protocol skeletons.

## Method and gates

* Kernels were authored at the TileLang source level, lowered with the exact
  pass configuration `tools/suite2_selected_worker.py` uses
  (`tl.disable_thread_storage_sync=True` for standalone sources and the
  CTA families; `{}` for the other templated families; the
  `InjectFenceProxy=identity` override for Proxy DATA_HAZARD), captured with
  `tools/tilelang_capture.py` (CPU-only, launch grid 1x1x1, sm_90a), and
  checked with `pipeguard check --frontend tilelang`.
* Gate (a) verdict: **all 105 tilelang cases match their label under the
  suite's authoritative engine `--engine maximal` (105/105 OK)**.
* Gate (b) structure: `tools/suite2_structural_audit.py --run-dir <recapture>`
  strict mode reports **21/21 tilelang cells PASS with 5/5 unique
  fingerprints** (recipe: op-type sequence + actor count + resource kinds).
* Gate (c) CUDA grounding: for every case the lowered `kernel.cu` was
  inspected; the per-case evidence line below shows the surviving construct
  (or, for repaired constructs, the case was redesigned - see
  "Lowering repairs encountered").
* Gate (d): each variant's scenario + rationale is listed below.
* Secondary reference: `--engine crosscheck` verdicts are recorded per case.
  Divergences (UNSUPPORTED / INCONCLUSIVE / different class) are *known
  crosscheck-engine limitations*, annotated inline and summarized at the end;
  the cells remain admitted on the maximal verdict per the project decision.

## Lowering repairs encountered (and how they were worked around)

1. **TileLang `thread_storage_sync` silently repairs source-level shared-memory
   races** whenever the templated pass config (`{}`) is used. Observed on:
   a cp.async missing-CTA-barrier candidate, a TMA store-without-barrier
   candidate, and four of five mbarrier DATA_HAZARD candidates (an injected
   `__syncthreads()` appears in `kernel.cu` and the checker then reports SAFE).
   Workarounds: (i) the **mbarrier DATA_HAZARD cell was converted to
   standalone sources** (standalone cases are lowered with
   `tl.disable_thread_storage_sync=True` by the worker, like the Memory and
   Barrier cells, so the protocol bug survives to CUDA); (ii) cp.async and
   TMA hazard variants were restricted to *async-completion* hazards
   (missing/misplaced `wait_group`, `tma_store_wait`, mbarrier waits), which an
   injected `__syncthreads` cannot repair - confirmed by grepping the lowered
   CUDA (the injected barrier is present but the verdict stays DATA_HAZARD
   because the async group is still in flight).
2. **Statically-false guards are eliminated by lowering** (used deliberately:
   the dead-code-sync variants' `kernel.cu` contains *no* barrier at all,
   which is exactly the CUDA-level hazard being labeled).
3. **InjectFenceProxy**: Proxy DATA_HAZARD variants rely on the runner's
   existing `InjectFenceProxy=identity` override; explicit
   `T.fence_proxy_async()` calls survive it (verified in `kernel.cu`).
4. **Barrier-init `__syncthreads`**: every `T.alloc_barrier` lowers to an
   elected `bar.init` + `fence_barrier_init` + one `__syncthreads` *before*
   the protocol body (a different pass from thread_storage_sync; emitted even
   with it disabled). It precedes all protocol operations, so it does not
   order the labeled conflicting accesses - the `syncthreads in CUDA: 1`
   entries on mbarrier-cell hazards are this init barrier, not a repair
   (e.g. `s2x-tilelang-data-hazard-mbarrier-04` lowers to
   `__syncthreads(); S[tx] = Data[tx]; Out[tx] = S[31-tx];` - the dead-guarded
   arrive/wait pair is gone and the handoff is bare).

## Checker-behavior findings hit during the redesign (candidates dropped)

* `wait_group(k>0)` under-wait and commit-of-empty-group candidates are
  classified DATA_HAZARD (not SYNC_HAZARD) by the checker - reassigned to the
  DATA_HAZARD cell or dropped.
* A *mis-placed* `fence_proxy_async` (fence before the generic write it should
  publish) is still credited by the checker (SAFE): fence binding appears to be
  positional-agnostic in capture. The "fence present but wrong position"
  idiom is therefore not usable as a DATA_HAZARD variant; replaced by
  second-generation-unfenced.
* Two `fence_proxy_async` calls in one kernel fail capture
  ("one unresolved target"); multi-generation each-fenced SAFE idiom replaced.
* `wgmma` without any `warpgroup_wait` is SYNC_HAZARD (matches robustness
  `wgmma_mut_no_wait`), so "wait removed + operand fill" cannot serve as a
  DATA_HAZARD variant.
* Cross-thread overlapping cp.async WAW (both groups waited per-thread, then
  CTA barrier) is reported SAFE by the checker; dropped.

## Per-cell inventory

Columns: variant, scenario, maximal verdict, crosscheck verdict, structural
fingerprint (first 12 hex of the audit recipe), op count, CUDA evidence.


### tilelang | Memory | DATA_HAZARD  (redesigned)

* **v0** `s2r-memory-tilelang-data-hazard-00` - **cross-warp-global-WAW (kept)**
  * rationale: cross-warp global WAW without ordering
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `1d3b61ab433a`, 4 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:20: Data[8] = 101;`
  * CUDA evidence: `kernel.cu:23: Data[8] = 102;`
* **v1** `s2r-memory-tilelang-data-hazard-01` - **dead-code-sync**
  * rationale: shared handoff whose only barrier sits under a statically false guard (tx<0)
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `c5269529e4bb`, 4 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
  * CUDA evidence: `kernel.cu:20: S[((int)threadIdx.x)] = Data[((int)threadIdx.x)];`
* **v2** `s2r-memory-tilelang-data-hazard-02` - **named-barrier-wrong-scope**
  * rationale: named partial barrier covers writer lanes 0..63 only; readers 64..127 are outside its participant set
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `f87666ed88fc`, 6 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::__sync_thread_partial(1, 64);`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
* **v3** `s2r-memory-tilelang-data-hazard-03` - **mbarrier-arrive-before-publish**
  * rationale: mbarrier arrival issued before the store it should publish; the wait does not order the handoff
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `e62dced3b5a2`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v4** `s2r-memory-tilelang-data-hazard-04` - **loop-same-phase-neighbor-race**
  * rationale: per-iteration barrier with correct parity, but R(S[127-tx]) and W(S[tx]) happen inside the same phase
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `6384eaf91223`, 23 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Memory | SAFE  (redesigned)

* **v0** `s2r-memory-tilelang-safe-00` - **disjoint-global-writes (kept)**
  * rationale: cross-warp disjoint global writes
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `1d3b61ab433a`, 4 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:20: Data[8] = 101;`
  * CUDA evidence: `kernel.cu:23: Data[12] = 102;`
* **v1** `s2r-memory-tilelang-safe-01` - **cta-barrier-handoff**
  * rationale: canonical shared handoff: write, full __syncthreads, cross-thread read
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `ed69140bd72a`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:21: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
* **v2** `s2r-memory-tilelang-safe-02` - **redundant-double-barrier**
  * rationale: over-synchronized handoff: two consecutive __syncthreads; FSM must not be confused
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `9b9ab53cf254`, 8 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:21: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
* **v3** `s2r-memory-tilelang-safe-03` - **named-barrier-exact-scope**
  * rationale: named partial barrier covers exactly the 64 lanes that exchange data
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `f87666ed88fc`, 6 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::__sync_thread_partial(1, 64);`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
* **v4** `s2r-memory-tilelang-safe-04` - **elected-mbarrier-publication**
  * rationale: expected-count-1 mbarrier armed by a single elected lane publishing S[0] to all waiters
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `8aa3ff1d955a`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Memory | SYNC_HAZARD  (redesigned)

* **v0** `s2r-memory-tilelang-sync-hazard-00` - **divergent-cta-barrier (kept)**
  * rationale: global RAW handoff guarded by a divergent CTA barrier
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `40867b8d9def`, 5 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:23: __syncthreads();`
  * CUDA evidence: `kernel.cu:20: Data[8] = 101;`
* **v1** `s2r-memory-tilelang-sync-hazard-01` - **mbarrier-missing-arrivals**
  * rationale: mbarrier expects 128 arrivals but only lanes 0..63 arrive: phase 0 never completes
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `8aa3ff1d955a`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v2** `s2r-memory-tilelang-sync-hazard-02` - **named-barrier-underfilled-scope**
  * rationale: named barrier declared for 64 participants but only 32 lanes reach it: it can never complete
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `f87666ed88fc`, 6 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::__sync_thread_partial(1, 64);`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int S[];`
* **v3** `s2r-memory-tilelang-sync-hazard-03` - **loop-constant-parity-overshoot**
  * rationale: mbarrier reused across 4 loop rounds but every wait uses parity 0; round 2 overshoots the phase account
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `6384eaf91223`, 23 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v4** `s2r-memory-tilelang-sync-hazard-04` - **second-phase-missing-arrival**
  * rationale: two-phase handoff: phase 0 completes, but the parity-1 wait has no second-generation arrivals
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `4dc81932d9f2`, 12 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:30: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Barrier | DATA_HAZARD  (spot-fixed variant [4])

* **v0** `s2r-barrier-tilelang-data-hazard-00` - **cross-warp global RAW handoff without or**
  * rationale: cross-warp global RAW handoff without ordering
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `70382443e55f`, 3 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
  * CUDA evidence: `kernel.cu:21: shared[0] = 101;`
* **v1** `s2r-barrier-tilelang-data-hazard-01` - **cross-warp global RAW handoff without or**
  * rationale: cross-warp global RAW handoff without ordering
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `1d3b61ab433a`, 4 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
  * CUDA evidence: `kernel.cu:21: shared[0] = 101;`
* **v2** `s2r-barrier-tilelang-data-hazard-02` - **cross-warp global RAW handoff without or**
  * rationale: cross-warp global RAW handoff without ordering
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `3b8631e767b8`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:23: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
* **v3** `s2r-barrier-tilelang-data-hazard-03` - **cross-warp global RAW handoff without or**
  * rationale: cross-warp global RAW handoff without ordering
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `413aa1e3cdc5`, 67 PGIR ops; syncthreads in CUDA: 0, proxy fences: 0
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
  * CUDA evidence: `kernel.cu:21: shared[0] = 1;`
* **v4** `s2r-barrier-tilelang-data-hazard-04` - **barrier-after-the-handoff**
  * rationale: a full __syncthreads exists but only after both the write and the cross-warp read already happened
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `d96eebe57fed`, 5 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:26: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Barrier | SAFE  (spot-fixed variant [4])

* **v0** `s2r-barrier-tilelang-safe-00` - **cross-warp global RAW handoff**
  * rationale: cross-warp global RAW handoff
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `40867b8d9def`, 5 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:23: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
* **v1** `s2r-barrier-tilelang-safe-01` - **cross-warp global RAW handoff**
  * rationale: cross-warp global RAW handoff
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `c9ce45413565`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:23: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
* **v2** `s2r-barrier-tilelang-safe-02` - **cross-warp global RAW handoff**
  * rationale: cross-warp global RAW handoff
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `5a086c2d4574`, 10 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:23: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
* **v3** `s2r-barrier-tilelang-safe-03` - **cross-warp global RAW handoff**
  * rationale: cross-warp global RAW handoff
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `1ae823e022ff`, 69 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:26: __syncthreads();`
  * CUDA evidence: `kernel.cu:19: extern __shared__ __align__(1024) int shared[];`
* **v4** `s2r-barrier-tilelang-safe-04` - **two-phase-mbarrier-parity-flip**
  * rationale: mbarrier reused for two handoff rounds with parity 0 then 1: generation accounting must track both
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `08b352336141`, 13 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:30: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Barrier | SYNC_HAZARD  (redesigned)

* **v0** `s2r-barrier-tilelang-sync-hazard-00` - **divergent-cta-barrier (kept)**
  * rationale: divergent CTA barrier
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `de8654286169`, 3 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:20: __syncthreads();`
  * CUDA evidence: `kernel.cu:23: Out[0] = 101;`
* **v1** `s2r-barrier-tilelang-sync-hazard-01` - **duplicate-divergent-arrival**
  * rationale: full __syncthreads then a second __syncthreads executed by lane 0 only: barrier divergence
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `536df4710ac5`, 5 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:19: __syncthreads();`
  * CUDA evidence: `kernel.cu:22: Out[1] = 102;`
* **v2** `s2r-barrier-tilelang-sync-hazard-02` - **circular-wait-two-mbarriers**
  * rationale: group A waits bar1 then arrives bar2; group B waits bar2 then arrives bar1: deadlock cycle
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `fb9bcd9331b4`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:30: __syncthreads();`
  * CUDA evidence: `kernel.cu:38: Out[2] = 103;`
* **v3** `s2r-barrier-tilelang-sync-hazard-03` - **elected-arrive-wrong-expected-count**
  * rationale: mbarrier expects 128 arrivals but only lane 0 arrives: phase 0 never completes
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `87f1ea235572`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v4** `s2r-barrier-tilelang-sync-hazard-04` - **wait-before-any-arrival**
  * rationale: all lanes wait parity 0 before any lane arrives: the arrivals are unreachable
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `cd9a5a6e1d27`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | mbarrier | DATA_HAZARD  (redesigned)

* **v0** `s2x-tilelang-data-hazard-mbarrier-00` - **arrive-before-publish**
  * rationale: arrival issued before the store it should publish; wait(parity 0) does not order W(S[tx]) before R(S[31-tx])
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `aacf3d775ff2`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v1** `s2x-tilelang-data-hazard-mbarrier-01` - **consumer-missing-wait**
  * rationale: producer arrives after publishing, but the consumer reads S[31] without waiting on the mbarrier
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `61cba24b9861`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:30: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v2** `s2x-tilelang-data-hazard-mbarrier-02` - **loop-same-phase-neighbor-race**
  * rationale: per-round mbarrier with correct parity, but R(S[31-tx]) and W(S[tx]) share one phase
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `d5990879093a`, 15 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v3** `s2x-tilelang-data-hazard-mbarrier-03` - **second-round-read-before-wait**
  * rationale: round 0 is fully ordered; in round 1 the read of R[31] happens before the parity-1 wait
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `55e43cdd78d6`, 13 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:30: __syncthreads();`
* **v4** `s2x-tilelang-data-hazard-mbarrier-04` - **dead-guarded-protocol**
  * rationale: the arrive/wait pair sits under a statically false guard (tx<0): the handoff is unordered
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `c718fbd6d74e`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
  * CUDA evidence: `kernel.cu:23: extern __shared__ __align__(1024) int S[];`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | mbarrier | SAFE  (unchanged (good cell))

* **v0** `s2x-tilelang-safe-mbarrier-00` - **MBARRIER:MISSING-ARRIVAL**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `8bee9537c48e`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v1** `s2x-tilelang-safe-mbarrier-01` - **MBARRIER:NEXT-ARRIVE-BEFORE-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `b0274252942c`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v2** `s2x-tilelang-safe-mbarrier-02` - **MBARRIER:WRONG-PARITY-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `afee818a7912`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v3** `s2x-tilelang-safe-mbarrier-03` - **MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `7061750bd535`, 141 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v4** `s2x-tilelang-safe-mbarrier-04` - **MBARRIER:LATE-MISSING-ARRIVAL**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `f8807ea92d50`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | mbarrier | SYNC_HAZARD  (unchanged (good cell))

* **v0** `s2x-tilelang-sync-hazard-mbarrier-00` - **MBARRIER:MISSING-ARRIVAL**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `a044d096b16c`, 5 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v1** `s2x-tilelang-sync-hazard-mbarrier-01` - **MBARRIER:NEXT-ARRIVE-BEFORE-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `d45dd1425940`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v2** `s2x-tilelang-sync-hazard-mbarrier-02` - **MBARRIER:WRONG-PARITY-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `afee818a7912`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
* **v3** `s2x-tilelang-sync-hazard-mbarrier-03` - **MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `d58e12a45035`, 140 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:31: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:28: __syncthreads();`
* **v4** `s2x-tilelang-sync-hazard-mbarrier-04` - **MBARRIER:LATE-MISSING-ARRIVAL**
  * rationale: legacy templated variant, kept unchanged
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `b0274252942c`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:29: bar[0].arrive();`
  * CUDA evidence: `kernel.cu:27: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | cp.async | DATA_HAZARD  (redesigned)

* **v0** `s2x-tilelang-data-hazard-cp-async-00` - **generic-store-into-inflight-destination**
  * rationale: kept legacy idiom: generic store S[0] issued between commit and wait_group(0); the cp.async write to the same bytes is still in flight (async-vs-generic WAW).
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `4f6f1e4dfa99`, 10 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:28: tl::cp_async_wait<0>();`
* **v1** `s2x-tilelang-data-hazard-cp-async-01` - **commit-without-wait**
  * rationale: cp.async committed but never waited; the following __syncthreads does NOT complete async-group writes, so the cross-thread read races the in-flight write (robustness `cpasync_missing_wait`).
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `01c8c7449ba9`, 6 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:25: __syncthreads();`
* **v2** `s2x-tilelang-data-hazard-cp-async-02` - **insufficient-wait-depth**
  * rationale: wait_group(1) with exactly one committed group retires nothing; the read of the destination races the in-flight write.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `3e0934382e21`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:24: tl::cp_async_wait<1>();`
* **v3** `s2x-tilelang-data-hazard-cp-async-03` - **two-groups-wait-one-touch-newest**
  * rationale: two committed groups; wait_group(1) retires only the older; a generic store to the NEWEST group's destination races its in-flight write.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `a35fac553f72`, 13 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:26: tl::cp_async_wait<1>();`
* **v4** `s2x-tilelang-data-hazard-cp-async-04` - **overlapping-destinations-one-group**
  * rationale: two cp.asyncs in one commit group write overlapping shared bytes [8,16): unordered async WAW inside the group.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `e333d481993c`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:25: tl::cp_async_wait<0>();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | cp.async | SAFE  (redesigned)

* **v0** `s2x-tilelang-safe-cp-async-00` - **issue-commit-wait-then-touch**
  * rationale: kept legacy idiom: wait_group(0) completes the group before the generic store and read.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `b257b046e9c4`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:24: tl::cp_async_wait<0>();`
* **v1** `s2x-tilelang-safe-cp-async-01` - **two-groups-wait-one-touch-retired**
  * rationale: wait_group(1) retires the older group; only the RETIRED group's destination is touched before wait_group(0).
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `a35fac553f72`, 13 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:26: tl::cp_async_wait<1>();`
* **v2** `s2x-tilelang-safe-cp-async-02` - **per-thread-copy-wait-plus-cta-barrier**
  * rationale: per-thread cp.async + wait_group(0) + __syncthreads; the barrier publishes every lane's completed copy for the cross-thread read (robustness `cpasync_ok`).
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `c7340fb59b2d`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:21: tl::cp_async_gs<4>((&(S[(((int)threadIdx.x) * 4)])), (&(A[(((int)threadIdx.x) * 4)])));`
  * CUDA evidence: `kernel.cu:23: tl::cp_async_wait<0>();`
* **v3** `s2x-tilelang-safe-cp-async-03` - **disjoint-destinations-one-group**
  * rationale: two cp.asyncs to disjoint shared ranges in one group, wait_group(0), read: control for DH v4.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `e333d481993c`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:25: tl::cp_async_wait<0>();`
* **v4** `s2x-tilelang-safe-cp-async-04` - **two-stage-prefetch-pipeline**
  * rationale: stage-k consumed under wait_group(1)/wait_group(0) ladder: the canonical prefetch pipeline protocol.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `309a4353600c`, 20 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:28: tl::cp_async_gs<16>((&(((uchar*)S)[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:32: tl::cp_async_wait<1>();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | cp.async | SYNC_HAZARD  (redesigned)

* **v0** `s2x-tilelang-sync-hazard-cp-async-00` - **wait-before-commit**
  * rationale: kept legacy idiom: wait_group(0) issued before any commit; the group it should cover does not exist yet.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `7251e59f78ca`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:23: tl::cp_async_wait<0>();`
* **v1** `s2x-tilelang-sync-hazard-cp-async-01` - **second-copy-never-committed**
  * rationale: copy issued after the last commit is never committed; wait_group(0) cannot cover it (group-accounting violation).
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `c42f9ee613fe`, 8 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:25: tl::cp_async_wait<0>();`
* **v2** `s2x-tilelang-sync-hazard-cp-async-02` - **divergent-mbarrier-completion**
  * rationale: cp.async pipeline's cross-thread handoff uses an mbarrier where only 16 of 32 expected arrivals happen: phase never completes.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `60318a7ea24c`, 12 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:30: tl::cp_async_gs<4>((&(S[(((int)threadIdx.x) * 4)])), (&(A[(((int)threadIdx.x) * 4)])));`
  * CUDA evidence: `kernel.cu:32: tl::cp_async_wait<0>();`
* **v3** `s2x-tilelang-sync-hazard-cp-async-03` - **commit-after-wait**
  * rationale: the commit that should make the group observable is issued only after the wait (mirror of wait-before-commit).
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `93071e95edb1`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:22: tl::cp_async_gs<16>((&(S[0])), (&(A[0])));`
  * CUDA evidence: `kernel.cu:23: tl::cp_async_wait<0>();`
* **v4** `s2x-tilelang-sync-hazard-cp-async-04` - **divergent-commit**
  * rationale: commit_group executed only by lanes <16: half the issued copies stay uncommitted while every lane waits.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `c7340fb59b2d`, 7 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:21: tl::cp_async_gs<4>((&(S[(((int)threadIdx.x) * 4)])), (&(A[(((int)threadIdx.x) * 4)])));`
  * CUDA evidence: `kernel.cu:25: tl::cp_async_wait<0>();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | TMA | DATA_HAZARD  (redesigned)

* **v0** `s2x-tilelang-data-hazard-tma-00` - **destination-read-before-wait**
  * rationale: kept legacy idiom: S read before mbarrier_wait_parity while the TMA load is in flight.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `cb7c3b990240`, 11 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: tl::tma_load((&(S[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:34: bar[0].arrive();`
* **v1** `s2x-tilelang-data-hazard-tma-01` - **wait-removed**
  * rationale: barrier armed and arrived but mbarrier_wait_parity removed: the read is not ordered after the async TMA write (robustness `tma_wait_removed`).
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `caf26d7333a9`, 10 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: tl::tma_load((&(S[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:34: bar[0].arrive();`
* **v2** `s2x-tilelang-data-hazard-tma-02` - **store-source-overwritten-before-wait**
  * rationale: tma store issued; the shared source is overwritten before tma_store_wait(read): WAR against the async read.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `6a7884b4c20c`, 11 PGIR ops; syncthreads in CUDA: 2, proxy fences: 1
  * CUDA evidence: `kernel.cu:27: tl::tma_store((&(Out[0])), (&(S[0])), 32);`
  * CUDA evidence: `kernel.cu:28: tl::tma_store_arrive();`
* **v3** `s2x-tilelang-data-hazard-tma-03` - **store-destination-read-before-wait**
  * rationale: global destination of the tma store read back before tma_store_wait: RAW against the async write.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `1aeb6cb318a2`, 10 PGIR ops; syncthreads in CUDA: 1, proxy fences: 1
  * CUDA evidence: `kernel.cu:27: tl::tma_store((&(Out[0])), (&(S[0])), 32);`
  * CUDA evidence: `kernel.cu:28: tl::tma_store_arrive();`
* **v4** `s2x-tilelang-data-hazard-tma-04` - **double-buffer-wrong-barrier-wait**
  * rationale: two loads on two barriers; only buffer0's barrier is waited but buffer1 is read.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `63cbaddbd4d8`, 15 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:37: tl::tma_load((&(((uchar*)S)[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:39: bar[0].arrive();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | TMA | SAFE  (redesigned)

* **v0** `s2x-tilelang-safe-tma-00` - **canonical-load**
  * rationale: transfer + arrive + wait_parity + read: the canonical mbarrier-completed TMA load.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `f700bcc7539d`, 11 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: tl::tma_load((&(S[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:34: bar[0].arrive();`
* **v1** `s2x-tilelang-safe-tma-01` - **two-transfers-one-barrier**
  * rationale: two transfers complete through one barrier (combined transaction count), then read.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `bc21c4275f95`, 13 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:34: tl::tma_load((&(((uchar*)S)[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:38: bar[0].arrive();`
* **v2** `s2x-tilelang-safe-tma-02` - **store-chain-wait-read**
  * rationale: write, __syncthreads, tma store, tma_store_wait(read=True), then source reuse: canonical publication chain.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `8ab0df502496`, 11 PGIR ops; syncthreads in CUDA: 2, proxy fences: 1
  * CUDA evidence: `kernel.cu:27: tl::tma_store((&(Out[0])), (&(S[0])), 32);`
  * CUDA evidence: `kernel.cu:28: tl::tma_store_arrive();`
* **v3** `s2x-tilelang-safe-tma-03` - **two-stores-one-barrier-one-wait**
  * rationale: two tma stores after one CTA barrier with a single tma_store_wait (fence/wait coverage regression control).
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `4dcc86c56d1a`, 12 PGIR ops; syncthreads in CUDA: 1, proxy fences: 1
  * CUDA evidence: `kernel.cu:30: tl::tma_store((&(Out[0])), (&(((uchar*)S)[0])), 32);`
  * CUDA evidence: `kernel.cu:31: tl::tma_store_arrive();`
* **v4** `s2x-tilelang-safe-tma-04` - **load-compute-store-pipeline**
  * rationale: load -> wait -> per-thread compute -> __syncthreads -> store -> wait: full pipeline.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `d7d316512c03`, 17 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:34: tl::tma_load((&(((uchar*)S)[0])), (&(A[0])), bar[0], 32);`
  * CUDA evidence: `kernel.cu:44: tl::tma_store_arrive();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | TMA | SYNC_HAZARD  (redesigned)

* **v0** `s2x-tilelang-sync-hazard-tma-00` - **expect-tx-overfill**
  * rationale: kept legacy idiom: mbarrier_expect_tx arms 48 bytes but the TMA delivers 32: the phase account can never balance.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `250b1c4f9bc1`, 12 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:33: tl::tma_load((&(shared[0])), (&(A[0])), barrier[0], 32);`
  * CUDA evidence: `kernel.cu:29: __syncthreads();`
* **v1** `s2x-tilelang-sync-hazard-tma-01` - **two-transfers-one-expected**
  * rationale: two transfers but only one transfer's bytes armed: over-delivery violates the transaction account.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `ef48c141cec1`, 14 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:33: tl::tma_load((&(shared[0])), (&(A[0])), barrier[0], 32);`
  * CUDA evidence: `kernel.cu:29: __syncthreads();`
* **v2** `s2x-tilelang-sync-hazard-tma-02` - **missing-arrive**
  * rationale: transfer lands but no thread ever arrives on the barrier: phase 0 never completes.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `c52b4cffd4fc`, 10 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: tl::tma_load((&(shared[0])), (&(A[0])), barrier[0], 32);`
  * CUDA evidence: `kernel.cu:29: __syncthreads();`
* **v3** `s2x-tilelang-sync-hazard-tma-03` - **expect-armed-no-arrive**
  * rationale: expect_tx armed and satisfied by the transfer, but the thread-arrival half of the phase is missing.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `4845dd606807`, 11 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:33: tl::tma_load((&(shared[0])), (&(A[0])), barrier[0], 32);`
  * CUDA evidence: `kernel.cu:29: __syncthreads();`
* **v4** `s2x-tilelang-sync-hazard-tma-04` - **second-phase-never-arrives**
  * rationale: phase 0 completes; a parity-1 wait follows with no second-generation arrival (late missing arrival).
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=SYNC_HAZARD
  * fingerprint `0f069f5222c6`, 12 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:32: tl::tma_load((&(shared[0])), (&(A[0])), barrier[0], 32);`
  * CUDA evidence: `kernel.cu:29: __syncthreads();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | WGMMA | DATA_HAZARD  (redesigned)

* **v0** `s2x-tilelang-data-hazard-wgmma-00` - **operand-overwrite-before-wait**
  * rationale: kept legacy idiom: operand A refilled before warpgroup_wait(0) while the gemm may still read it.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `3a33a3a378c5`, 65 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v1** `s2x-tilelang-data-hazard-wgmma-01` - **shallow-wait-operand-overwrite**
  * rationale: two gemms; warpgroup_wait(1) leaves one in flight; operand refilled anyway.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `343a18835678`, 70 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v2** `s2x-tilelang-data-hazard-wgmma-02` - **tma-prefetch-into-live-operand**
  * rationale: next-tile TMA prefetch writes operand A before warpgroup_wait(0): async write vs in-flight gemm read.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `de024c476273`, 65 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v3** `s2x-tilelang-data-hazard-wgmma-03` - **accumulator-clear-before-wait**
  * rationale: accumulator cleared between the gemm and its warpgroup_wait: WAW with the in-flight gemm write.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `9615999dd346`, 67 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v4** `s2x-tilelang-data-hazard-wgmma-04` - **operand-refill-between-gemms**
  * rationale: operand B refilled between two gemms with no intervening wait: WAR against the first gemm.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `1dc10cb2bb02`, 71 PGIR ops; syncthreads in CUDA: 4, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | WGMMA | SAFE  (redesigned)

* **v0** `s2x-tilelang-safe-wgmma-00` - **gemm-wait-copy**
  * rationale: kept legacy idiom: gemm, warpgroup_wait(0), copy accumulator out.
  * maximal=SAFE (label SAFE, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `30c92b238056`, 59 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v1** `s2x-tilelang-safe-wgmma-01` - **operand-reuse-after-wait**
  * rationale: operand refilled only after warpgroup_wait(0): correctly ordered reuse.
  * maximal=SAFE (label SAFE, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `e8c36c5a97dc`, 65 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v2** `s2x-tilelang-safe-wgmma-02` - **batched-two-gemms-one-wait**
  * rationale: two gemms retired by a single warpgroup_wait(0) before the copy.
  * maximal=SAFE (label SAFE, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `3f0ea940c886`, 62 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v3** `s2x-tilelang-safe-wgmma-03` - **shallow-then-full-wait**
  * rationale: warpgroup_wait(1) then warpgroup_wait(0) before operand reuse: staged retirement.
  * maximal=SAFE (label SAFE, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `c87d1926f901`, 70 PGIR ops; syncthreads in CUDA: 3, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v4** `s2x-tilelang-safe-wgmma-04` - **two-rounds-wait-each**
  * rationale: two gemm rounds each closed by warpgroup_wait(0) (accumulator reuse across rounds).
  * maximal=SAFE (label SAFE, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `40e05f3c22ed`, 64 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | WGMMA | SYNC_HAZARD  (redesigned)

* **v0** `s2x-tilelang-sync-hazard-wgmma-00` - **copy-before-wait**
  * rationale: kept legacy idiom: accumulator copied out before the warpgroup_wait that retires the gemm.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `d3c67e912bc3`, 59 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v1** `s2x-tilelang-sync-hazard-wgmma-01` - **wait-removed**
  * rationale: no warpgroup_wait at all after the gemm (robustness `wgmma_mut_no_wait`).
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `87c1a25db695`, 57 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v2** `s2x-tilelang-sync-hazard-wgmma-02` - **wait-before-gemm**
  * rationale: warpgroup_wait placed before the gemm it should cover: nothing to retire, gemm never completed.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `101633431b79`, 59 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:58: tl::warpgroup_wait<0>();`
* **v3** `s2x-tilelang-sync-hazard-wgmma-03` - **copy-between-gemms**
  * rationale: accumulator copied out between two gemms; the only wait comes after.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `805cd2021af0`, 62 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v4** `s2x-tilelang-sync-hazard-wgmma-04` - **shallow-wait-before-copy**
  * rationale: two gemms but only warpgroup_wait(1) before the copy: the newest gemm still writes the accumulator.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `3f0ea940c886`, 62 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:62: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Proxy | DATA_HAZARD  (redesigned)

* **v0** `s2x-tilelang-data-hazard-proxy-00` - **unfenced-generic-to-async-store**
  * rationale: kept legacy idiom: generic fill then tma store with no fence_proxy_async (InjectFenceProxy disabled by the runner).
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `d4ff5dde4a62`, 9 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:30: tl::tma_store((&(Out[0])), (&(shared[0])), 8192);`
  * CUDA evidence: `kernel.cu:31: tl::tma_store_arrive();`
* **v1** `s2x-tilelang-data-hazard-proxy-01` - **second-generation-unfenced**
  * rationale: generation 1 fenced and stored; generation 2 refills and stores with NO new fence.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `9400ffe7ea01`, 21 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:31: tl::tma_store((&(Out[0])), (&(shared[0])), 8192);`
  * CUDA evidence: `kernel.cu:32: tl::tma_store_arrive();`
* **v2** `s2x-tilelang-data-hazard-proxy-02` - **generic-fill-during-tma-load**
  * rationale: generic fill of the buffer while a tma load into the same buffer is in flight (async-vs-generic WAW across proxies).
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `eb4fa0f12caa`, 21 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:34: tl::tma_load((&(((half_t*)shared)[0])), (&(Out[0])), bar[0], 8192);`
  * CUDA evidence: `kernel.cu:36: bar[0].arrive();`
* **v3** `s2x-tilelang-data-hazard-proxy-03` - **generic-global-write-unfenced-load**
  * rationale: generic global write, then a tma load reads that global with no fence: cross-proxy RAW on the global side.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `22fdbb8d9589`, 18 PGIR ops; syncthreads in CUDA: 2, proxy fences: 0
  * CUDA evidence: `kernel.cu:37: tl::tma_load((&(((half_t*)shared)[0])), (&(Out[0])), bar[0], 8192);`
  * CUDA evidence: `kernel.cu:39: bar[0].arrive();`
* **v4** `s2x-tilelang-data-hazard-proxy-04` - **fragment-staged-unfenced-store**
  * rationale: accumulator staged via generic copy (fragment->shared) then tma store with no fence.
  * maximal=DATA_HAZARD (label DATA_HAZARD, OK); crosscheck=DATA_HAZARD
  * fingerprint `383cf234b4db`, 21 PGIR ops; syncthreads in CUDA: 1, proxy fences: 0
  * CUDA evidence: `kernel.cu:35: tl::tma_store((&(Out[0])), (&(shared[0])), 8192);`
  * CUDA evidence: `kernel.cu:36: tl::tma_store_arrive();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Proxy | SAFE  (redesigned)

* **v0** `s2x-tilelang-safe-proxy-00` - **fence-store-wait**
  * rationale: kept legacy idiom: fill, fence_proxy_async, tma store, tma_store_wait.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `f236c62668fb`, 10 PGIR ops; syncthreads in CUDA: 1, proxy fences: 1
  * CUDA evidence: `kernel.cu:31: tl::tma_store((&(Out[0])), (&(shared[0])), 8192);`
  * CUDA evidence: `kernel.cu:32: tl::tma_store_arrive();`
* **v1** `s2x-tilelang-safe-proxy-01` - **fence-then-load-overwrite**
  * rationale: fill, fence, tma load overwriting the same buffer, mbarrier wait, read: ordered cross-proxy WAW.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `846bb2718e58`, 24 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:41: tl::tma_load((&(((half_t*)shared)[0])), (&(Out[0])), bar[0], 8192);`
  * CUDA evidence: `kernel.cu:43: bar[0].arrive();`
* **v2** `s2x-tilelang-safe-proxy-02` - **one-fence-two-stores**
  * rationale: one fence publishes two generically filled buffers to two tma stores.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `7d6885e091aa`, 12 PGIR ops; syncthreads in CUDA: 1, proxy fences: 1
  * CUDA evidence: `kernel.cu:38: tl::tma_store((&(OutFlat[0])), (&(((half_t*)s1)[0])), 4096);`
  * CUDA evidence: `kernel.cu:39: tl::tma_store_arrive();`
* **v3** `s2x-tilelang-safe-proxy-03` - **generic-reuse-after-store-wait**
  * rationale: fence + store + tma_store_wait(read) then generic reuse of the source buffer.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `2377d2b6d071`, 20 PGIR ops; syncthreads in CUDA: 2, proxy fences: 1
  * CUDA evidence: `kernel.cu:33: tl::tma_store((&(Out[0])), (&(((half_t*)shared)[0])), 8192);`
  * CUDA evidence: `kernel.cu:34: tl::tma_store_arrive();`
* **v4** `s2x-tilelang-safe-proxy-04` - **load-compute-fence-store-roundtrip**
  * rationale: tma load -> mbarrier wait -> generic copy -> fence -> tma store -> wait: full round trip.
  * maximal=SAFE (label SAFE, OK); crosscheck=SAFE
  * fingerprint `957c9b69d365`, 23 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:34: tl::tma_load((&(((half_t*)shared)[0])), (&(Out[0])), bar[0], 8192);`
  * CUDA evidence: `kernel.cu:47: tl::tma_store_arrive();`
  
  Cell fingerprint uniqueness: 5/5 distinct.

### tilelang | Proxy | SYNC_HAZARD  (redesigned)

* **v0** `s2x-tilelang-sync-hazard-proxy-00` - **fenced-gemm-copy-before-wait**
  * rationale: kept legacy idiom: fenced operand, gemm, accumulator copied out before the final warpgroup_wait.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `7b1f2aec28ab`, 66 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:72: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v1** `s2x-tilelang-sync-hazard-proxy-01` - **fenced-gemm-no-wait**
  * rationale: fenced operand and gemm, but no warpgroup_wait at all.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `2ba1202405ab`, 64 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:72: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v2** `s2x-tilelang-sync-hazard-proxy-02` - **wait-before-fenced-gemm**
  * rationale: warpgroup_wait before the gemm it should retire.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `b108054d416e`, 66 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:68: tl::warpgroup_wait<0>();`
* **v3** `s2x-tilelang-sync-hazard-proxy-03` - **copy-between-fenced-gemms**
  * rationale: accumulator copy between two gemms; wait only at the end.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `c383e9485e92`, 69 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:72: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
* **v4** `s2x-tilelang-sync-hazard-proxy-04` - **shallow-wait-before-copy**
  * rationale: two gemms with only warpgroup_wait(1) before the accumulator copy.
  * maximal=SYNC_HAZARD (label SYNC_HAZARD, OK); crosscheck=UNSUPPORTED  (crosscheck divergence: UNSUPPORTED; admitted on maximal per project decision)
  * fingerprint `c0e05bb6b1a4`, 69 PGIR ops; syncthreads in CUDA: 3, proxy fences: 1
  * CUDA evidence: `kernel.cu:43: tl::tma_load(A_desc, barrier_a[0], (&(((half_t*)A_shared)[0])), 0, 0);`
  * CUDA evidence: `kernel.cu:72: tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)A_shared)[0])));`
  
  Cell fingerprint uniqueness: 5/5 distinct.

## Crosscheck divergence summary

| case | label (maximal verdict) | crosscheck |
|---|---|---|
| s2r-memory-tilelang-sync-hazard-03 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-mbarrier-02 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-data-hazard-wgmma-00 | DATA_HAZARD | UNSUPPORTED |
| s2x-tilelang-data-hazard-wgmma-01 | DATA_HAZARD | UNSUPPORTED |
| s2x-tilelang-data-hazard-wgmma-02 | DATA_HAZARD | UNSUPPORTED |
| s2x-tilelang-data-hazard-wgmma-03 | DATA_HAZARD | UNSUPPORTED |
| s2x-tilelang-data-hazard-wgmma-04 | DATA_HAZARD | UNSUPPORTED |
| s2x-tilelang-safe-wgmma-00 | SAFE | UNSUPPORTED |
| s2x-tilelang-safe-wgmma-01 | SAFE | UNSUPPORTED |
| s2x-tilelang-safe-wgmma-02 | SAFE | UNSUPPORTED |
| s2x-tilelang-safe-wgmma-03 | SAFE | UNSUPPORTED |
| s2x-tilelang-safe-wgmma-04 | SAFE | UNSUPPORTED |
| s2x-tilelang-sync-hazard-wgmma-00 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-wgmma-01 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-wgmma-02 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-wgmma-03 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-wgmma-04 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-proxy-00 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-proxy-01 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-proxy-02 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-proxy-03 | SYNC_HAZARD | UNSUPPORTED |
| s2x-tilelang-sync-hazard-proxy-04 | SYNC_HAZARD | UNSUPPORTED |

Diagnostic codes behind these divergences:

* All 15 WGMMA cases and the 5 Proxy SYNC_HAZARD cases (wgmma pipelines) fail
  the crosscheck expansion with `INVALID_ASYNC_ENGINE` ("explicit engine
  ordering must reference an ASYNC_ENGINE actor"). This is pre-existing: the
  legacy wgmma/proxy cells were already UNSUPPORTED under crosscheck before
  this redesign (verified on the legacy `s2x-tilelang-sync-hazard-wgmma-00`).
* `s2r-memory-tilelang-sync-hazard-03` (constant-parity loop) and the legacy
  `s2x-tilelang-sync-hazard-mbarrier-02` (wrong-parity wait, untouched cell)
  fail with `MALFORMED_STREAM_PARITY` ("raw wait parity does not match its
  logical generation") - the same behavior the robustness pool shows for
  `loop_parity_constant` / `mbar_wrong_parity`.

These are crosscheck-engine work items, not label errors; every case above is
correctly classified by the authoritative maximal engine.

## Gate 2 (tools/suite2_cuda_grounding.py) rule updates required

The built-in rule table of `tools/suite2_cuda_grounding.py` encodes the
*legacy single-idiom* shape of each tilelang family (e.g. `tilelang:TMA:*`
requires `tl::tma_load` + `.expect_transaction(` + `.wait(`, and
`tilelang:Memory:DATA_HAZARD` forbids any `__syncthreads`). With five
deliberately different skeletons per cell these family-level patterns no
longer hold: a wait-removed hazard legitimately has no wait, store-side TMA
variants have no `tma_load`, and several hazards legitimately contain a
non-repairing barrier (too late / wrong scope / barrier-init). Against the
default table the recapture reports 12 FAIL + 5 NEEDS_REVIEW, every one of
which is a rule-table artifact, not a grounding failure (the per-case CUDA
evidence is in the inventory above).

With the family-intersection override below (`--config`), the gate reports
**105/105 PASS** on the same recapture:

```json
{
 "schema_version": "pipeguard-suite2-cuda-grounding-config-v1",
 "rules": {
  "tilelang:Memory:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "kernel_entry", "pattern": "__global__\\s+void", "strict": true}
   ]
  },
  "tilelang:Memory:SYNC_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "sync_construct", "pattern": "__syncthreads\\s*\\(|__sync_thread_partial\\(|\\.wait\\(", "strict": true}
   ]
  },
  "tilelang:Barrier:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "kernel_entry", "pattern": "__global__\\s+void", "strict": true}
   ]
  },
  "tilelang:TMA:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "tma_op", "pattern": "tl::tma_(load|store)\\(", "strict": true}
   ]
  },
  "tilelang:TMA:SAFE": {
   "forbidden": [],
   "required": [
    {"name": "tma_op", "pattern": "tl::tma_(load|store)\\(", "strict": true},
    {"name": "completion_wait", "pattern": "\\.wait\\(|tl::tma_store_wait", "strict": true}
   ]
  },
  "tilelang:TMA:SYNC_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "tma_load", "pattern": "tl::tma_load\\(", "strict": true},
    {"name": "mbarrier_init", "pattern": "\\.init\\(", "strict": true},
    {"name": "mbarrier_wait", "pattern": "\\.wait\\(", "strict": true}
   ]
  },
  "tilelang:cp.async:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "cp_async_issue", "pattern": "tl::cp_async_gs", "strict": true},
    {"name": "cp_async_commit", "pattern": "tl::cp_async_commit|cp\\.async\\.commit_group", "strict": true}
   ]
  },
  "tilelang:cp.async:SYNC_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "cp_async_issue", "pattern": "tl::cp_async_gs", "strict": true},
    {"name": "cp_async_wait", "pattern": "tl::cp_async_wait|cp\\.async\\.wait_group", "strict": true}
   ]
  },
  "tilelang:mbarrier:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "mbarrier_init", "pattern": "\\.init\\(", "strict": true}
   ]
  },
  "tilelang:Proxy:DATA_HAZARD": {
   "forbidden": [],
   "required": [
    {"name": "tma_op", "pattern": "tl::tma_(load|store)\\(", "strict": true}
   ]
  },
  "tilelang:Proxy:SAFE": {
   "forbidden": [],
   "required": [
    {"name": "proxy_fence", "pattern": "tl::fence_proxy_async", "strict": true},
    {"name": "tma_op", "pattern": "tl::tma_(load|store)\\(", "strict": true}
   ]
  }
 }
}
```

Note the override is intentionally weaker at the family level; per-variant
grounding is established by the per-case evidence lines plus the capture
snapshot, per ADMISSION-GATES.md. Tools are out of this redesign's write
scope, so the override is recorded here for the maintainer to adopt (either
as `--config` input or folded into the built-in table).

## File-level change summary

* Rewritten standalone sources (lowered with `tl.disable_thread_storage_sync`):
  * `memory/{data_hazard,safe,sync_hazard}/*-0[1-4].py` (12 cases; variant 0 kept per cell)
  * `barrier/sync_hazard/*-0[1-4].py` (4 cases; variant 0 kept)
  * `barrier/data_hazard/*-04.py`, `barrier/safe/*-04.py` (fingerprint-collision spot fixes)
  * `mbarrier/data_hazard/*.py` (5 cases, converted from templated TMA-OBSERVE to standalone
    mbarrier-protocol kernels; their `.json` specs no longer carry `template_family`)
* Rewritten templated builders in `suite/frontend_pairs/sources/tilelang_cases.py`
  (variant-dispatched, five distinct skeletons per class; case `.py`/`.json`
  files of these cells unchanged):
  * `_suite2_cp_async_case`, `_suite2_wgmma_case`, `_suite2_proxy_case`,
    `_suite2_tma_sync_case`, new `_suite2_tma_diverse_case`
    (now routed from family `TMA-OBSERVE` in `build_tilelang_case`).
* NOT touched: `suite/suite2/manifest.json`, `FILES.sha256`,
  `selection.lock.json`, `cases/triton/**`, `tools/**`. Manifest source/spec
  sha256 entries for the changed files are stale by design and need the
  maintainer's re-hash + recapture pass.

## Maintainer follow-ups (outside this redesign's write scope)

1. `suite/suite2/manifest.json`: refresh `source`/`spec` sha256+bytes for the
   46 changed files under `cases/tilelang/` (and re-run the recapture to
   refresh pgir/checker evidence entries). `FILES.sha256` likewise.
2. `tools/suite2_selected_worker.py` / `tools/suite2_nvidia_worker.py`
   argument tables (used only when kernels are actually launched; the
   capture path does not launch): the redesigned kernel signatures are
   - primitive `cp.async` (all classes): `A:(128,) uint8`, `Out:(32,) uint8`,
     `out_idx=1` (the nvidia worker's `(9 + variant) * 16` comment/sizing at
     `tools/suite2_nvidia_worker.py:157` refers to the removed ladder);
   - family `TMA-OBSERVE` and `TMA-SYNC`: `A:(64,) uint8`, `Out:(64,)`/`(1,)`
     `uint8`, `out_idx=1` (no longer `16 * (variant + 1)` wide);
   - Proxy SAFE variant 2: single output `OutFlat:(4096,) float16`
     (other Proxy store variants keep `Out:(64, 64) float16`), `out_idx=0`;
   - mbarrier DATA_HAZARD (now standalone): `kernel(Data:(256,) int32,
     Out:(32,) int32)`, `out_idx=1`.
3. `tools/suite2_cuda_grounding.py`: adopt the family-rule overrides above.

