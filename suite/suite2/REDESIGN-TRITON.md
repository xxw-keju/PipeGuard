# Suite 2 Triton Redesign: structural diversity rework

Date: 2026-10-27. Scope: the Triton half of the 210-case labeled suite.

## Why

An audit of the previous Triton cases found that within most
(dsl, primitive, result_class) cells the five "variants" shared a single
protocol-event-sequence skeleton (pure shape duplicates) or differed only by a
repeat count (arithmetic ladders).  This rework gives every redesigned cell five
structurally DISTINCT synchronization idioms.  The structural fingerprint used
throughout is the sha256 of the comma-joined PGIR operation-type sequence from
the capture snapshot (`snapshot["operations"][*]["op"]`), with shapes/masks/
counts normalized away -- i.e. two cases are duplicates iff their protocol
event-type sequences coincide.

## Design principles applied

1. Every label is grounded at the TTGIR level: each construct was compiled with
   the pinned Triton (61c02bcd) to post-pipeline TTGIR and the hazard/safety
   verified to survive compilation (several draft variants were rejected
   because DCE/CSE removed or merged the construct -- see Findings).
2. Within a cell the five variants are five distinct protocol-event-sequence
   skeletons (fingerprints verified distinct; intentional cross-result-class
   pairs share skeletons, which is desirable polarity pairing).
3. Labels follow the CUDA memory model: same-address unordered concurrent
   writes are DATA_HAZARD even when they write the same value
   (s2r-memory-triton-data-hazard-03 encodes this rule directly).

## Engines and acceptance gate

Per the project lead's correction, the authoritative engine for this suite is
`maximal` (tools/run_suite2_selected210.py runs `check_module(engine="maximal")`,
and the unified H100 rerun uses the same path).  The acceptance gate for every
case below is therefore: maximal verdict == independent label.  `crosscheck`
results are recorded as a secondary reference only: across the 105 cases it
agrees on 55, and on the remaining 50 it returns a non-verdict outcome
(INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH x20, UNSUPPORTED/
COLLECTIVE_ARRIVAL_COUNT_MISMATCH x10 on divergent-barrier cases, UNSUPPORTED/
GROUP_OWNER_MISMATCH x10 on TMA-store/warp-specialized cases, UNSUPPORTED/
MALFORMED_STREAM_PARITY x3, and 7 timeouts >420s on wgmma-class snapshots) --
it never returns a verdict contradicting a label.

## Local validation methodology (no GPU, no torch)

Each case source was compiled to post-pipeline TTGIR with the triton-pg
interpreter via `GluonASTSource` + `triton.compile` (MockTensor-based argument
binding reproducing `tools/suite2_selected_worker.py:triton_prepare` exactly,
including the Proxy DATA_HAZARD `add_proxy_fence_insertion=noop` compiler
override and the final-PTX evidence for WGMMA/Proxy-SYNC capture).  The TTGIR
was captured with `tools/triton_capture.py` through the worker's own
`triton_capture_with_evidence` (same binding evidence construction as the
unified run) and checked with the checker environment via
`python -m pipeguard check <snapshot> --frontend triton` under both the
`maximal` engine (the engine used by tools/run_suite2_selected210.py, i.e. the
ground-truth engine for the suite) and the `crosscheck` engine.

Because the unified H100 run recaptures every case through `kernel.warmup`
with real torch tensors, every redesigned case is additionally marked
VALIDATION_PENDING_H100; by construction the signatures, constexprs and
launch parameters match `triton_prepare`, so the H100 TTGIR is expected to be
identical to the locally compiled TTGIR.


## triton|Memory|SAFE  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-memory-triton-safe-00
- idiom: disjoint WAW: concurrent writes land on disjoint 4-byte intervals
- PGIR skeleton: `mem.write,mem.write`
- fingerprint: `1dbd4acd785cd43f9881545dbf34d6c7439131807a02e603e4ea0a04f3bc80e4`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-safe-01
- idiom: same-thread RAW: program order within one thread sequences write before read
- PGIR skeleton: `mem.write,mem.read,mem.write`
- fingerprint: `4cdbf5abdc577ce131d7c4c7b03df05982dcd11d55a62c1d7568aafa57c5c85a`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-safe-02
- idiom: adjacent-disjoint WAR: read and write touch adjacent but non-overlapping elements
- PGIR skeleton: `mem.read,mem.write,mem.write`
- fingerprint: `f8f0f34753a9dabb458385d8f2323732cbb66dcf68d9081021202633b67ba8d5`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-safe-03
- idiom: barrier-ordered handoff: full CTA barrier provides the RAW ordering edge
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `928c56ae30da1d07d9fff3e1c832cb77c8b958e6cbbb2995934231cde45ae5de`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-safe-04
- idiom: same-thread double-buffer alternation: one thread ping-pongs between two slots in program order
- PGIR skeleton: `mem.write,mem.read,mem.write,mem.read,mem.write`
- fingerprint: `57dba7e7be5955a3d0015bc608d133d05fe3e6a198ce84480a6bcd0c517e1e89`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Memory|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-memory-triton-data-hazard-00
- idiom: cross-warp WAW: two unordered writes of distinct values to one global address
- PGIR skeleton: `mem.write,mem.write`
- fingerprint: `1dbd4acd785cd43f9881545dbf34d6c7439131807a02e603e4ea0a04f3bc80e4`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-data-hazard-01
- idiom: cross-warp RAW: reader observes writer's address with no ordering edge
- PGIR skeleton: `mem.write,mem.read,mem.write`
- fingerprint: `4cdbf5abdc577ce131d7c4c7b03df05982dcd11d55a62c1d7568aafa57c5c85a`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-data-hazard-02
- idiom: cross-warp WAR: write races ahead of an unordered concurrent read
- PGIR skeleton: `mem.read,mem.write,mem.write`
- fingerprint: `f8f0f34753a9dabb458385d8f2323732cbb66dcf68d9081021202633b67ba8d5`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-data-hazard-03
- idiom: same-value WAW plus third-party reader: CUDA memory model treats unordered same-address writes as a race even when values agree
- PGIR skeleton: `mem.write,mem.write,mem.read,mem.write`
- fingerprint: `e6b6337f92614e2c85f7baa8df6cbf74169ef4b7bdcb56cc67ec5315d747963f`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-data-hazard-04
- idiom: loop-carried cross-thread RAW: unrolled two-iteration producer/consumer with no per-iteration sync
- PGIR skeleton: `mem.write,mem.read,mem.write,mem.write,mem.read,mem.write`
- fingerprint: `b4f64531880bf7897fb5e8a6b8c92a0f476bd45112fac33a81dee3a5319f49ac`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Memory|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-memory-triton-sync-hazard-00
- idiom: RAW guarded only by a divergent CTA barrier (64 of 128 lanes arrive) [kept divergent-barrier-RAW idiom of previous variant 0]
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `928c56ae30da1d07d9fff3e1c832cb77c8b958e6cbbb2995934231cde45ae5de`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-sync-hazard-01
- idiom: WAW separated only by a divergent CTA barrier (32 of 128 lanes arrive)
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.write`
- fingerprint: `f12332d27ec5104f0dea283347b70fff61c9edbb51846df5aa6d97a4ad00eb72`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-sync-hazard-02
- idiom: WAR guarded only by a divergent CTA barrier (96 of 128 lanes arrive)
- PGIR skeleton: `mem.read,barrier.arrive,barrier.observe,mem.write,mem.write`
- fingerprint: `cb2f4ce0f29a7e7d802bc08ae82d47f0e235bf1a9cb78371c456e393bc54d3ce`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-sync-hazard-03
- idiom: correct full-CTA round followed by a divergent second round (second-phase divergence)
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `057e8d5a6f5384eb53aecaf3c8e386a98ebd41b5b6f72ac3966e78d315f09842`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-memory-triton-sync-hazard-04
- idiom: divergent barrier before a producer round: stuck lanes can never reach the following full CTA barrier
- PGIR skeleton: `barrier.arrive,barrier.observe,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `bb443d1f385c03df3fc6751aa989a76f28036e2c286bba1bb9c83ba9f6c24b70`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Barrier|DATA_HAZARD  (kept, audit-good)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-barrier-triton-data-hazard-00
- idiom: BARRIER:SHARED-RAW
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `928c56ae30da1d07d9fff3e1c832cb77c8b958e6cbbb2995934231cde45ae5de`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-data-hazard-01
- idiom: BARRIER:SHARED-WAW
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `a48732404e67f00f4d9d46bcebab1e977b2016cc30aa7c0d374a734242cad377`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-data-hazard-02
- idiom: BARRIER:SHARED-WAR
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write,mem.write`
- fingerprint: `fbb6231c45b67eddc0e0c8dec34eb7809e3b4e637b3a602d348fbaf159bc137a`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-data-hazard-03
- idiom: BARRIER:SHARED-DELAYED-RAW
- PGIR skeleton: `mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `c0bef6ac23cc439ef95c674299c9ff0943fd4aab6b6d232d7d8510a7cf00afcd`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-data-hazard-04
- idiom: BARRIER:GLOBAL-RAW
- PGIR skeleton: `mem.write,mem.read,mem.write`
- fingerprint: `4cdbf5abdc577ce131d7c4c7b03df05982dcd11d55a62c1d7568aafa57c5c85a`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Barrier|SAFE  (kept, audit-good; variant 04 replaced in follow-up)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-barrier-triton-safe-00
- idiom: BARRIER:SHARED-RAW
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `928c56ae30da1d07d9fff3e1c832cb77c8b958e6cbbb2995934231cde45ae5de`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-safe-01
- idiom: BARRIER:SHARED-WAW
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `7cd61c0cbec3ec6ca3e3048dc2c8a732edf81ed812cc50fad1989cf12cf2dff6`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-safe-02
- idiom: BARRIER:SHARED-WAR
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write,barrier.arrive,barrier.observe,mem.write`
- fingerprint: `03b544f0f1d24a36ad01d14f0392d3371c441690270efd94f6f1e2c6a9956b03`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-safe-03
- idiom: BARRIER:SHARED-DELAYED-RAW
- PGIR skeleton: `mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `c0bef6ac23cc439ef95c674299c9ff0943fd4aab6b6d232d7d8510a7cf00afcd`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-safe-04
- idiom: two-way request/response handoff ordered by two full CTA barrier rounds
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.read,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `7dfd9020b48b696efce96e183a939c4a7b5d10228ba7088b500569dffcbb4016`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Barrier|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2r-barrier-triton-sync-hazard-00
- idiom:  [kept from previous suite (anchor variant)]
- PGIR skeleton: `barrier.arrive,barrier.observe,mem.write`
- fingerprint: `cecff29ea2f6345daad9465081cf3b858f3fef34380f343ea21e95117abac631`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-sync-hazard-01
- idiom: complete CTA round followed by a divergent round: only the second rendezvous is broken
- PGIR skeleton: `barrier.arrive,barrier.observe,barrier.arrive,barrier.observe,mem.write`
- fingerprint: `0866d70c86434b66005e76f7f983902876a9b98f9e949dde48338f170e313d91`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-sync-hazard-02
- idiom: shared-memory handoff whose only ordering edge is a divergent barrier
- PGIR skeleton: `mem.read,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `8f855a36ae38296a9f7d7b50c488f5543154df6cf1fda33df8c3b879b1eb5184`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-sync-hazard-03
- idiom: divergent barrier interposed between two shared-memory write rounds (WAW republish)
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,mem.write,mem.write`
- fingerprint: `122241e3f807b080f3effc97fa070c111c821c702c3dfc79b129a6c84d42dbfb`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2r-barrier-triton-sync-hazard-04
- idiom: three-round handshake ring whose middle rendezvous is divergent (full -> divergent -> full)
- PGIR skeleton: `barrier.arrive,barrier.observe,mem.write,barrier.arrive,barrier.observe,mem.read,barrier.arrive,barrier.observe,mem.write`
- fingerprint: `a3517a648743ac34422dbe95dc9be6da6a14e4b94ca28a863c8d38f20795a450`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|cp.async|SAFE  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-safe-cp-async-00
- idiom: CP-ASYNC:SINGLE-GROUP-ROUNDTRIP
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `1c08c9dc854cef6c15f84ce7e0730f18c29f6ea05bda2fdb6cf003f36835f797`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-cp-async-01
- idiom: CP-ASYNC:STAGED-WAIT-DEPTHS
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write,cp_async.wait_group,mem.read,mem.write`
- fingerprint: `263efad9d6145ab2bb6da39330f34865319940d2d6bf6185e38ca7691dee33f4`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-cp-async-02
- idiom: CP-ASYNC:TWO-COPIES-ONE-GROUP
- PGIR skeleton: `cp_async.issue,cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.read,mem.write`
- fingerprint: `4c4eae57a3842eb32b7ed071d15ee98bbb10ccdaac71a8af4712ead53439fbdf`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-cp-async-03
- idiom: CP-ASYNC:REUSE-AFTER-WAIT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.wait_group,mem.write,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `2f407c4f13eb05ce3ebd30382a86c8a850c1dade2dafdd9eb364c217b3b0ab24`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-cp-async-04
- idiom: CP-ASYNC:LOOKAHEAD-ONE-PIPELINE
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,cp_async.wait_group,mem.read,mem.write`
- fingerprint: `87974362d22fdb9c7d929594324a978b5d6dd60bb9d790086a68224d67a43965`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|cp.async|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-data-hazard-cp-async-00
- idiom: CP-ASYNC:MISSING-WAIT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `77968da7220bd44a160e66fd57b631eda13910b9a178c8e0eca193b6488d5ce7`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-cp-async-01
- idiom: CP-ASYNC:LAX-WAIT-ONE-IN-FLIGHT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `1c08c9dc854cef6c15f84ce7e0730f18c29f6ea05bda2fdb6cf003f36835f797`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-cp-async-02
- idiom: CP-ASYNC:GENERIC-OVERWRITE-IN-FLIGHT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,mem.write,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `52edc2d7460bde992b7ff9119493d6095d8d4ef646622934a97f5298af4b71bc`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-cp-async-03
- idiom: CP-ASYNC:OVERLAPPING-ASYNC-DESTINATIONS
- PGIR skeleton: `cp_async.issue,cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `b47e28aaac4536ce8bc71bd7c0bcc090df1eb2b724fda7bc5806b56c02255a65`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-cp-async-04
- idiom: CP-ASYNC:WRONG-BUFFER-UNDER-STAGED-WAIT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `b906a30c5907935f5d723a1193e057edcc66564d1b184fc7611696cb53de3ca7`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|cp.async|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-sync-hazard-cp-async-00
- idiom: CP-ASYNC:WAIT-BEFORE-COMMIT [kept wait-before-commit scenario (previous cell scenario), minimal form]
- PGIR skeleton: `cp_async.issue,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `209746239f5e9ffec6386aa698f34666622cf463e697b195f092cea17134cf7b`
- maximal: SYNC_HAZARD/WAIT_BEFORE_COMMIT  | crosscheck: SYNC_HAZARD/WAIT_BEFORE_COMMIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-cp-async-01
- idiom: CP-ASYNC:UNCOMMITTED-COPY-DANGLING
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.issue,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `e03c951ff45c02e0fa82a900bb11db6b3504f4390640c3fc6044715b9b45e833`
- maximal: SYNC_HAZARD/WAIT_BEFORE_COMMIT  | crosscheck: SYNC_HAZARD/WAIT_BEFORE_COMMIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-cp-async-02
- idiom: CP-ASYNC:WAIT-SPLITS-ISSUE-FROM-COMMIT
- PGIR skeleton: `cp_async.issue,cp_async.wait_group,cp_async.commit_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `d0bdc3cdc34c1f789dd75bb5a2d174cdf83e8c388898620690a14622449eaf38`
- maximal: SYNC_HAZARD/WAIT_BEFORE_COMMIT  | crosscheck: SYNC_HAZARD/WAIT_BEFORE_COMMIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-cp-async-03
- idiom: CP-ASYNC:SECOND-ROUND-WAIT-BEFORE-COMMIT
- PGIR skeleton: `cp_async.issue,cp_async.commit_group,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,cp_async.issue,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `67c5fb412c10080e00a9479a744f1fb055a5eb632b00aec58a1588f024986926`
- maximal: SYNC_HAZARD/WAIT_BEFORE_COMMIT  | crosscheck: SYNC_HAZARD/WAIT_BEFORE_COMMIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-cp-async-04
- idiom: CP-ASYNC:BARE-WAIT-EMPTY-STREAM
- PGIR skeleton: `cp_async.wait_group,cp_async.issue,cp_async.wait_group,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `6e8ba58581be64e0756e7feb6da2f0b5dae3b7c845cdc3425897e09ad91a3330`
- maximal: SYNC_HAZARD/WAIT_BEFORE_COMMIT  | crosscheck: SYNC_HAZARD/WAIT_BEFORE_COMMIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|TMA|SAFE  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-safe-tma-00
- idiom: TMA-LOAD:CANONICAL-EXPECT-ISSUE-WAIT [kept canonical expect/issue/wait skeleton (previous variant 0)]
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `b978ee433cecbf77c54b8ed6213a57f2ab2db503cee38d19a9d7769e20608086`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-tma-01
- idiom: TMA-LOAD:ONE-EXPECT-TWO-TRANSFERS
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,tma_load.issue,mbarrier.wait_parity,mem.read,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `708e3c339c3a96582a5c7991a825b188094d83c0c198d12726e4b9a6fe346c7b`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-tma-02
- idiom: TMA-LOAD:TWO-ROUND-PARITY-FLIP
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `14d4dc46d3ab16d23de8d0ce056f69110a1ec3123f50728cdeee3ec2dee2fcff`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-tma-03
- idiom: TMA-LOAD:FENCED-GENERIC-PROLOGUE
- PGIR skeleton: `mbarrier.init,mem.write,mbarrier.arrive_expect_tx,proxy_fence,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `76af7978b45fe9ad4cd0b55f229483404cc21a02d3cd45d24fd744a1875b78c2`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-tma-04
- idiom: TMA-LOAD:PER-SLOT-BARRIERS
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.arrive_expect_tx,mbarrier.arrive_expect_tx,tma_load.issue,tma_load.issue,mbarrier.wait_parity,mbarrier.wait_parity,mem.read,mem.read,mem.write,mbarrier.invalidate,mbarrier.invalidate`
- fingerprint: `f7b127e94f4fdeccf70d83475e2b6384fe1c3872bcf355c39d367a1369416fc4`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|TMA|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-data-hazard-tma-00
- idiom: TMA-LOAD:DESTINATION-BEFORE-WAIT [kept read-before-wait skeleton (previous variant 0)]
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mem.read,mbarrier.wait_parity,mem.write,mbarrier.invalidate`
- fingerprint: `a799c2e5738cc5a4a7274638c60696f54c92f1ecf35bf6f5dd2aaa7c7da40564`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-tma-01
- idiom: TMA-LOAD:MISSING-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mem.read,mem.write`
- fingerprint: `8b894d3905005443ae3e827ba200853969f38f98abe153ef3bfc5426b998f205`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-tma-02
- idiom: TMA-LOAD:READ-BETWEEN-ISSUES
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mem.read,tma_load.issue,mbarrier.wait_parity,mem.write,mbarrier.invalidate`
- fingerprint: `d56929b46532a9aa483d8a2f5f425fba3ea6ca0d0e279ad76f25f229a1dfa2ba`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-tma-03
- idiom: TMA-LOAD:GENERIC-WRITE-IN-FLIGHT-DEST
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mem.write,mbarrier.wait_parity,barrier.arrive,barrier.observe,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `3408cd3fd8c66fbfcd751fef5ca94888658e0592f76238b985d421eb96aa130c`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-tma-04
- idiom: TMA-LOAD:SECOND-ROUND-MISSING-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mbarrier.arrive_expect_tx,tma_load.issue,mem.read,mem.write`
- fingerprint: `f01fafc9ad3ff517d36e7171734557117dafb7243a05edc828de851aab19a2d4`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|TMA|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-sync-hazard-tma-00
- idiom: TMA-SYNC:OVER-EXPECT-NEVER-COMPLETES [kept over-expect skeleton (previous variant 0)]
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `b978ee433cecbf77c54b8ed6213a57f2ab2db503cee38d19a9d7769e20608086`
- maximal: SYNC_HAZARD/TX_BYTES_MISMATCH  | crosscheck: SYNC_HAZARD/TX_BYTES_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-tma-01
- idiom: TMA-SYNC:UNDER-EXPECT-OVERDELIVERY
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,tma_load.issue,mbarrier.wait_parity,mem.read,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `708e3c339c3a96582a5c7991a825b188094d83c0c198d12726e4b9a6fe346c7b`
- maximal: SYNC_HAZARD/TX_BYTES_MISMATCH  | crosscheck: SYNC_HAZARD/STALE_GENERATION
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-tma-02
- idiom: TMA-SYNC:STALE-PARITY-SECOND-ROUND
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `14d4dc46d3ab16d23de8d0ce056f69110a1ec3123f50728cdeee3ec2dee2fcff`
- maximal: SYNC_HAZARD/WRONG_PARITY  | crosscheck: UNSUPPORTED/MALFORMED_STREAM_PARITY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-tma-03
- idiom: TMA-SYNC:DOUBLE-EXPECT-ACCUMULATION
- PGIR skeleton: `mbarrier.init,mbarrier.arrive_expect_tx,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate`
- fingerprint: `0fac65fdcf243689fa1869f4f2238c152e8d7b7004f18e7ed9e81683f5283cfc`
- maximal: SYNC_HAZARD/INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE  | crosscheck: SYNC_HAZARD/ARRIVE_WITH_ZERO_PENDING_COUNT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-tma-04
- idiom: TMA-SYNC:WAIT-ON-WRONG-BARRIER
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.arrive_expect_tx,tma_load.issue,mbarrier.wait_parity,mem.read,mem.write,mbarrier.invalidate,mbarrier.invalidate`
- fingerprint: `a48f8e21861be90b19716a339828c9d72223b713b427a44e981fdc1e9d202c14`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: SYNC_HAZARD/STUCK_WAIT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|mbarrier|SAFE  (kept, audit-good; variant 04 replaced in follow-up)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-safe-mbarrier-00
- idiom: MBARRIER:MISSING-ARRIVAL
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `2d891bbe72ebec42b2289db8171cbb6bbaad660a9135fb0a5abbe02e0fe375b0`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-mbarrier-01
- idiom: MBARRIER:NEXT-ARRIVE-BEFORE-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `b8966a2cde0dde350547195d86c76c832a8abcb4acd65dd0f6f6e9ca274525fe`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-mbarrier-02
- idiom: MBARRIER:WRONG-PARITY-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `fc160e3a4f8a6d83b5e4c0a5b5b6eef4bf5690414df5ae2dd4bc32f0ac9b52c8`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-mbarrier-03
- idiom: MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT
- PGIR skeleton: `mbarrier.init,mem.write,mbarrier.arrive,mbarrier.wait_parity,barrier.arrive,barrier.observe,mem.read,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mbarrier.arrive,mbarrier.wait_parity,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `d664d68d7205c15f269c4b57f9975e0db6f3d2979ea1e31b7ae338e718cac12b`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-mbarrier-04
- idiom: MBARRIER:OBSERVED-INVALIDATE-REINIT-LIFECYCLE
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.invalidate,mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.invalidate`
- fingerprint: `51ea2fa3630f11c1219518a94db01328592bcdd1086053e8a65a2968430fc329`
- maximal: SAFE/  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|mbarrier|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-data-hazard-mbarrier-00
- idiom: MBARRIER:ARRIVE-BEFORE-PUBLISH
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.wait_parity,mem.read,mem.write,mbarrier.arrive,mem.write`
- fingerprint: `4648e8cd0dba5efb2016d788a0133a0a44458a8a4a86fb4536ea9fab79eb9151`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-mbarrier-01
- idiom: MBARRIER:MISSING-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.init,mem.read,mem.write,mem.write,mbarrier.arrive`
- fingerprint: `568dce9c33ae31d297de8f78717e472a9e6c4bbea1cc584e04083e1660f273cd`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-mbarrier-02
- idiom: MBARRIER:WRONG-BARRIER-SELF-RELEASED
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mem.read,mem.write,mem.write,mbarrier.arrive`
- fingerprint: `e3550163107eba1dbefa27bfa5db1e1e8f70e80c9a25f4e123b4e649ffbdc3c1`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-mbarrier-03
- idiom: MBARRIER:GENERATION-SKEW-READ-BEFORE-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.wait_parity,mem.read,mbarrier.arrive,mem.read,mbarrier.wait_parity,mem.write,mem.write,mbarrier.arrive,mbarrier.wait_parity,mem.write,mbarrier.arrive`
- fingerprint: `a8da87e5ec94585bd4dd85c644f4f64357ac5fd0ddb8ee5ba7433a3a2df61157`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-mbarrier-04
- idiom: MBARRIER:UNPUBLISHED-SECOND-SLOT
- PGIR skeleton: `mbarrier.init,mbarrier.init,mbarrier.wait_parity,mem.read,mem.write,mem.write,mbarrier.arrive,mem.write`
- fingerprint: `0519fd1444a3af342722fe142a9a9ca7a930d0ea125a88e9ca3e2dc0506c7307`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|mbarrier|SYNC_HAZARD  (kept, audit-good)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-sync-hazard-mbarrier-00
- idiom: MBARRIER:MISSING-ARRIVAL
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `2d891bbe72ebec42b2289db8171cbb6bbaad660a9135fb0a5abbe02e0fe375b0`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: SYNC_HAZARD/MISSING_ARRIVAL
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-mbarrier-01
- idiom: MBARRIER:NEXT-ARRIVE-BEFORE-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.arrive,mbarrier.wait_parity,mbarrier.wait_parity`
- fingerprint: `bcbd0a55030e11d7e5fae4cab9a317a078c5b6d68a7a39b9c6b05ac097df1753`
- maximal: SYNC_HAZARD/INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE  | crosscheck: SYNC_HAZARD/INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-mbarrier-02
- idiom: MBARRIER:WRONG-PARITY-WAIT
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `fc160e3a4f8a6d83b5e4c0a5b5b6eef4bf5690414df5ae2dd4bc32f0ac9b52c8`
- maximal: SYNC_HAZARD/WRONG_PARITY  | crosscheck: UNSUPPORTED/MALFORMED_STREAM_PARITY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-mbarrier-03
- idiom: MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT
- PGIR skeleton: `mbarrier.init,mem.write,mbarrier.arrive,barrier.arrive,barrier.observe,mem.read,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mem.write,mbarrier.arrive,mbarrier.wait_parity,barrier.arrive,barrier.observe,mem.read,mem.write`
- fingerprint: `81480644f7060a719f1c5b68d1992b5f4e36952d3f0c50e57d84066e5ce9396b`
- maximal: SYNC_HAZARD/INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE  | crosscheck: UNSUPPORTED/MALFORMED_STREAM_PARITY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-mbarrier-04
- idiom: MBARRIER:LATE-MISSING-ARRIVAL
- PGIR skeleton: `mbarrier.init,mbarrier.arrive,mbarrier.wait_parity,mbarrier.arrive,mbarrier.wait_parity`
- fingerprint: `b8966a2cde0dde350547195d86c76c832a8abcb4acd65dd0f6f6e9ca274525fe`
- maximal: SYNC_HAZARD/STUCK_PROTOCOL_STATE  | crosscheck: SYNC_HAZARD/MISSING_ARRIVAL
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|WGMMA|SAFE  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-safe-wgmma-00
- idiom: WGMMA:CANONICAL-ASYNC-WAIT0 [kept canonical async+wait(0) skeleton (previous repeats=1)]
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `f353ce2ce2a96620fb2d8aa324547a2ac18d6dbbbacaa8b80d4823b601d676ea`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-wgmma-01
- idiom: WGMMA:TWO-GROUPS-EACH-CONSUMED-AFTER-WAIT0
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `0779959e9b5587d0e5a945a95e9ccdd1a400bd3fc5ef9623acfe772218c34053`
- maximal: SAFE/  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-wgmma-02
- idiom: WGMMA:DEPTH1-PIPELINE
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `9c22a1ba07c93267ef4793d87092cd4006f5d1e83ee4010c23b7d6fd3bf8bf02`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-wgmma-03
- idiom: WGMMA:OPERAND-REUSE-AFTER-WAIT
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.write,mem.read,mem.write`
- fingerprint: `cab029d0494a079eae47fb3b97bc07e1752b9dd675eb116467f6f52987a048bf`
- maximal: SAFE/  | crosscheck: SAFE/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-wgmma-04
- idiom: WGMMA:DUAL-ENDPOINT-REDUNDANT-WAIT
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `78b48aaabbfd0c9d6dac82578db4667708f8423a14a6f834ecbab934a47dcfa1`
- maximal: SAFE/  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|WGMMA|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-data-hazard-wgmma-00
- idiom: WGMMA:OPERAND-OVERWRITE-IN-FLIGHT [kept operand-overwrite-before-wait skeleton (previous mode)]
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `708cdb7e343cfb92ab8a4deb285596b0731d461c4fb4c2234a5898bab9389dc0`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-wgmma-01
- idiom: WGMMA:PENDING-NEWEST-OPERAND-OVERWRITE
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `e50c1130217817f935f0d95a9fad9e90f6beb09a24dd142938f3a51f8ee52192`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-wgmma-02
- idiom: WGMMA:BARRIER-DOES-NOT-ORDER-ASYNC-ENGINE
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,barrier.arrive,barrier.observe,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `4d89f8efccb5549c5fe0c8ef8b7eeeb9294441a977f5f5f77116541ce243f680`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-wgmma-03
- idiom: WGMMA:BOTH-OPERANDS-CLOBBERED-IN-FLIGHT
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,mem.write,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `3735f0e851afb6389cdf92697bb67f759f353e34fbae4f9f6b5c519f86728f1f`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: DATA_HAZARD/UNORDERED_CONFLICT
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-wgmma-04
- idiom: WGMMA:DEPTH3-PIPELINE-UNDERSYNC-ROTATION
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `84c75aba9b259ef69db7804aa69f78a878ed34064e9bb04fd356337d653a5bcf`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|WGMMA|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-sync-hazard-wgmma-00
- idiom: WGMMA:LAX-WAIT1-CONSUME [kept skeleton of previous _suite2_wgmma SYNC variant (repeats=1)]
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group`
- fingerprint: `d13a3fa6caad2799b58d6a6608117afb44c7819b0a0b7799c9b0574abe322f10`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-wgmma-01
- idiom: WGMMA:CHAINED-PIPELINE-CONSUME-UNDER-WAIT1
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `eeb076400950e13acc3136f133f979594d2b9ad549d5c578e305dbdc959dec62`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-wgmma-02
- idiom: WGMMA:TWO-GROUPS-NEWEST-CONSUMED-FIRST
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `0779959e9b5587d0e5a945a95e9ccdd1a400bd3fc5ef9623acfe772218c34053`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-wgmma-03
- idiom: WGMMA:LAX-WAIT2-SINGLE-GROUP
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `f353ce2ce2a96620fb2d8aa324547a2ac18d6dbbbacaa8b80d4823b601d676ea`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-wgmma-04
- idiom: WGMMA:GOOD-ROUND-THEN-LAX-ROUND
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `f8157efd063643dc4c019383fd1d1775c19ed1f07e1cbbf4059a8334f39f6696`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Proxy|SAFE  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-safe-proxy-00
- idiom: PROXY:WARP-SPECIALIZED-PUBLISH-FENCE-STORE [kept skeleton of previous warp-specialized variant]
- PGIR skeleton: `mem.write,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `7a62504216bf287dbbc051d9549f88c9a46dc9b10f77eb67e00e4edfc21411bb`
- maximal: SAFE/  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-proxy-01
- idiom: PROXY:MINIMAL-PUBLISH-STORE-ENDPOINT
- PGIR skeleton: `mem.write,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group`
- fingerprint: `1922f49e919027acf1b67523abc3d57d061b29dbc7b61ce4b18216472919e81f`
- maximal: SAFE/  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-proxy-02
- idiom: PROXY:ALTERNATING-TILE-TWO-ROUNDS
- PGIR skeleton: `mem.write,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.write,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `e5db7a310adcca035124bc574cc1ede8fe4c794b3b166c9985d646f53128ce8d`
- maximal: SAFE/  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-proxy-03
- idiom: PROXY:BARRIER-HANDOFF-PLUS-FENCE
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `57c6e8bef967522c6cedb59e4385417f21bce096bbecfa8b5caf2a678a37b4a5`
- maximal: SAFE/  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-safe-proxy-04
- idiom: PROXY:STAGED-STORE-WAITS
- PGIR skeleton: `mem.write,proxy_fence,tma_store.issue,tma_store.commit_group,tma_store.wait_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `096f080ca14ec7f88aae19b9e04dbf41b2b9e3f0a384e2cfaf1c83ad6547b59c`
- maximal: SAFE/  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Proxy|DATA_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-data-hazard-proxy-00
- idiom: PROXY:WARP-SPECIALIZED-PUBLISH-NO-FENCE [kept skeleton of previous warp-specialized variant]
- PGIR skeleton: `mem.write,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `f2d2421dbd092750ae90debe7817875696f24f4cdc2d594b8a3649b22e373d85`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-proxy-01
- idiom: PROXY:UNFENCED-MINIMAL-PUBLISH
- PGIR skeleton: `mem.write,tma_store.issue,tma_store.commit_group,tma_store.wait_group`
- fingerprint: `f8b1fe756a7525dbe9e7f5baa59628b8fd4b31748fcf42efc789f9354723cbf5`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-proxy-02
- idiom: PROXY:ALTERNATING-TILES-UNFENCED
- PGIR skeleton: `mem.write,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.write,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `58e49618c5c4b1538acfb3a3093524c4a85ebf2cb377e65c497e9f37d73916fd`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-proxy-03
- idiom: PROXY:BARRIER-WITHOUT-PROXY-FENCE
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,tma_store.issue,tma_store.commit_group,tma_store.wait_group,mem.read,mem.write`
- fingerprint: `66ed4dd61bbd5c7bd87f29bb6a56cf6858c04a711c9cb7b7618ca1b736cd05eb`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-data-hazard-proxy-04
- idiom: PROXY:CONSUME-OUTPUT-BEFORE-STORE-WAIT
- PGIR skeleton: `mem.write,tma_store.issue,tma_store.commit_group,mem.read,tma_store.wait_group,mem.write`
- fingerprint: `1d54844cecf796116c5c671a6894fa7f1f171643387e5b2cc6ec2bcc8f5b5ef7`
- maximal: DATA_HAZARD/UNORDERED_CONFLICT  | crosscheck: UNSUPPORTED/GROUP_OWNER_MISMATCH
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## triton|Proxy|SYNC_HAZARD  (redesigned)

Distinct fingerprints: 5/5; maximal verdict == label: 5/5

### s2x-triton-sync-hazard-proxy-00
- idiom: PROXY-WGMMA:LAX-WAIT1-CONSUME [kept skeleton of previous _suite2_wgmma SYNC variant (repeats=1)]
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group`
- fingerprint: `d13a3fa6caad2799b58d6a6608117afb44c7819b0a0b7799c9b0574abe322f10`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-proxy-01
- idiom: PROXY-WGMMA:EARLY-CONSUME-THEN-CORRECT-CONSUME
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `72742bfc320f7afe5e07be3143142afdecf90026e396f802ac9e0bc688b09886`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-proxy-02
- idiom: PROXY-WGMMA:FENCED-LAX-WAIT2-NO-TRAILING
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `f353ce2ce2a96620fb2d8aa324547a2ac18d6dbbbacaa8b80d4823b601d676ea`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: SYNC_HAZARD/WGMMA_RESULT_NOT_READY
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-proxy-03
- idiom: PROXY-WGMMA:TWO-GROUPS-CONSUME-NEWEST-FIRST
- PGIR skeleton: `mem.write,mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `0779959e9b5587d0e5a945a95e9ccdd1a400bd3fc5ef9623acfe772218c34053`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

### s2x-triton-sync-hazard-proxy-04
- idiom: PROXY-WGMMA:SINGLE-PUBLISH-GOOD-THEN-LAX-ROUND
- PGIR skeleton: `mem.write,barrier.arrive,barrier.observe,proxy_fence,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write,wgmma.fence_operand,wgmma.issue,wgmma.commit_group,wgmma.wait_group,wgmma.wait_group,mem.read,mem.write`
- fingerprint: `c8c5d2e3e947c78773b753c0c227a7268fc4972dbdc2834afb9fab21bbb3a19c`
- maximal: SYNC_HAZARD/WGMMA_RESULT_NOT_READY  | crosscheck: TIMEOUT_GT_420S/
- status: VERIFIED_LOCAL_TTGIR + VALIDATION_PENDING_H100

## Cell summary

| cell | redesigned | distinct fingerprints | maximal==label |
|---|---|---|---|
| triton|Memory|SAFE | yes | 5/5 | 5/5 |
| triton|Memory|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|Memory|SYNC_HAZARD | yes | 5/5 | 5/5 |
| triton|Barrier|DATA_HAZARD | no | 5/5 | 5/5 |
| triton|Barrier|SAFE | no | 5/5 | 5/5 |
| triton|Barrier|SYNC_HAZARD | yes | 5/5 | 5/5 |
| triton|cp.async|SAFE | yes | 5/5 | 5/5 |
| triton|cp.async|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|cp.async|SYNC_HAZARD | yes | 5/5 | 5/5 |
| triton|TMA|SAFE | yes | 5/5 | 5/5 |
| triton|TMA|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|TMA|SYNC_HAZARD | yes | 5/5 | 5/5 |
| triton|mbarrier|SAFE | no | 5/5 | 5/5 |
| triton|mbarrier|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|mbarrier|SYNC_HAZARD | no | 5/5 | 5/5 |
| triton|WGMMA|SAFE | yes | 5/5 | 5/5 |
| triton|WGMMA|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|WGMMA|SYNC_HAZARD | yes | 5/5 | 5/5 |
| triton|Proxy|SAFE | yes | 5/5 | 5/5 |
| triton|Proxy|DATA_HAZARD | yes | 5/5 | 5/5 |
| triton|Proxy|SYNC_HAZARD | yes | 5/5 | 5/5 |

## Residual duplicates in audit-good cells (fixed in follow-up)

Two audit-good cells initially retained one type-sequence duplicate pair each
(barrier|SAFE 00 vs 04, mbarrier|SAFE 01 vs 04).  Per the project lead's
follow-up instruction, the 04 case of each pair was replaced with a new,
structurally distinct SAFE idiom (two-round request/response barrier handoff;
observed-invalidate-reinit mbarrier lifecycle), restoring 5/5 fingerprint
uniqueness in both cells; the other variants of those cells are untouched.

## Findings recorded during construction

- **Reuse-after-store_wait is contested**: a generic shared write placed after
  `tma.store_wait(0)` on the same tile is reported DATA_HAZARD by the current
  checker even though `cp.async.bulk.wait_group.read 0` orders it on hardware;
  the repo's own `_named_tma_store_kernel` default branch reproduces this.
  The redesigned SAFE cells avoid the construct entirely so that no labeled
  case depends on the contested direction.
- **DCE/CSE must be re-checked for every WGMMA variant**: two
  `warpgroup_mma` calls with identical operands and identical zero-initialized
  accumulators are CSE-merged into one TTGIR dot, and a dot whose result is
  never consumed is dead-code-eliminated.  Two-group variants therefore use
  distinct accumulator seeds and consume every result.
- **Capture constraints discovered** (fail-closed boundaries that constrain
  expressible idioms): only the exact predicated `bar.sync 0` inline-asm
  pattern is recognized (one per kernel -- PTX register scoping forbids two);
  a `fence_async_shared` with no following async consumer is rejected;
  TMA loads without any preceding `barrier_expect` are rejected
  (TRITON_TMA* fail-closed); kernels whose capture uses final-PTX evidence
  (WGMMA primitive, Proxy SYNC_HAZARD) admit at most one PTX-level proxy
  fence, which rules out generic republish between wgmma groups.
- **mbarrier invalidate dominance**: `mbarrier.invalidate` on a barrier with
  an unobserved phase reclassifies a data-hazard case as
  SYNC_HAZARD/INVALIDATE_INCOMPLETE; redesigned DATA_HAZARD variants omit the
  invalidate when the protocol intentionally leaves a phase unobserved.
- **Phase-reuse UB detector**: a producer re-arming an mbarrier before the
  consumer observed the previous phase triggers
  INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE (SYNC);
  the generation-skew DATA_HAZARD variant therefore uses a reverse-handshake
  second barrier to re-arm legally and keeps only the early gen-2 read as the
  seeded defect.
- **crosscheck engine limits** (expected, not regressions): divergent-barrier
  cases -> UNSUPPORTED/COLLECTIVE_ARRIVAL_COUNT_MISMATCH; warp-specialized TMA
  stores -> UNSUPPORTED/GROUP_OWNER_MISMATCH; several TMA cells ->
  INCONCLUSIVE/REPRESENTATION_CROSSCHECK_MISMATCH; wgmma-class snapshots can
  exceed 400s.  The suite's ground-truth engine (maximal, as used by
  tools/run_suite2_selected210.py) is the match criterion above.

## Handoff notes for the unified H100 rerun

- `suite/suite2/manifest.json`, `FILES.sha256` and `selection.lock.json` were
  intentionally NOT touched (outside this task's write scope).
  `tools/run_suite2_selected210.py:normalize_case` verifies the manifest's
  per-case `source.sha256` / `spec.sha256`; those hashes now mismatch for the
  86 redesigned case files (17 cells x 5 minus the kept-as-is
  s2r-barrier-triton-sync-hazard-00, plus the follow-up fixes to
  s2r-barrier-triton-safe-04 and s2x-triton-safe-mbarrier-04), so the manifest
  must be regenerated (or its hash fields refreshed) before the unified rerun.
- `suite/frontend_pairs/sources/triton_cases.py` now imports torch lazily
  (inside `compile_triton_case`) so the module and the `_s2v2_*` factories are
  importable in torch-free environments; the H100 warmup path is unchanged.
- Every redesigned case keeps its original case_id, primitive, result_class,
  variant index, template_family and semantic_knobs, so
  `triton_prepare`/`triton_run` argument construction, binding-evidence
  retries, descriptor shapes and compiler overrides all apply unchanged.
- Local artifacts for review: per-case TTGIR + snapshots under
  /tmp/pgredo/runs-final/<case_id>/ (and /tmp/pgredo/runs*/ for iteration
  history) on the build machine, and the merged result ledger at
  /tmp/pgredo/results.{s1,s2,s3}.json.
