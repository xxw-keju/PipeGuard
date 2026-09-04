# Checker assurance boundary

PipeGuard separates three claims that must not be conflated: the NVIDIA-valid mbarrier core theorem, bounded implementation validation, and frontend extraction correctness.

## NVIDIA-valid core

The exact claim applies to finite, statically ranked, freshly initialized Arrive/LogicalWait programs with fixed participants and capacity, coalesced polling, resolved gap-free logical wait streams, and exact event identities. A logical wait targets physical generation `g`; it may complete only while `g` is the immediately preceding completed generation. Partial arrivals in generation `g + 1` remain legal. The checker rejects the phase-completing transition from `g + 1` to `g + 2` when a required observer of `g` remains.

Within this domain:

```text
checker returns RISK
iff
there is a reachable protocol-invalid attempt or valid stuck state
```

If no invalid attempt is reachable, the raw NVIDIA-validity model and the lifted exact-generation model have isomorphic rooted reachable transition systems. On SAFE programs, the direct Arrive/Wait PDG therefore represents the same must-order relation.

This is not a claim about arbitrary bit-only parity programs, dynamic barrier identities, unbounded control flow, unknown participants, unresolved polling identity, or incomplete frontend metadata. Those inputs fail closed.

## Independent bounded enumeration

`tests/test_bounded_core_oracle.py` implements a declarative oracle independent of `Executor.enabled`, `Executor.apply`, state fingerprints, held-anchor scheduling, and checker PDG construction. Its state is only:

```text
(actor PCs, current generation, pending count, generation-started bit,
 publications, Required/Remaining identities, executed identities)
```

The test exhaustively enumerates 460 static programs over capacities 1–3 and two or three actors:

| Class | Programs |
|---|---:|
| SAFE | 92 |
| Risk | 368 |
| └ phase/role-invalid | 346 |
| └ valid underfill stuck | 22 |

It checks all of the following:

1. maximal and memoized all-state checker results crosscheck, and their `SAFE`/`SYNC_HAZARD` verdict agrees with the independent raw oracle;
2. for all 114 programs with no reachable invalid attempt, raw and exact reachable state sets are equal;
3. for SAFE programs, all 2,274 ordered pairs of distinct program events agree between oracle must-order and checker PDG reachability;
4. capacities 1, 2, and 3 include the NVIDIA-valid partial-next-generation boundary and the final phase-completing guard;
5. underfilled barriers exercise stuck completeness separately from invalid-attempt completeness.

Finite enumeration is regression evidence for the implementation; it does not replace the quantified core proof or extend its domain.

## PDG and incremental work bound

Let:

```text
N_ext = program and finite synthetic transitions + explicit dependency references
T_eff = nonempty program, phase, collective, group, channel, and engine streams
```

The incremental scheduler performs one canonical pass and one stateful delayed-anchor pass per stream. Hash-table/set operations are expected-amortized constant time. Delta notifications touch indexed followers rather than scanning all transitions. The direct full-boundary graph has:

```text
O(N_ext) intrinsic protocol/order edges
+ at most O(T_eff * N_ext) emitted boundary edges
```

Thus scheduling plus direct PDG construction is expected-amortized `O(T_eff * N_ext)` plus the graph entries emitted. The statistics certificate records transition attempts, notification work, boundary evaluations, edge insertions, zero full enabled scans, zero state clones, and zero prefix replays. Tests use operation counts, not wall-clock ratios.

PDG adjacency is maintained at edge insertion time. Reachability computes and caches one closure per queried source after the graph is frozen, with explicit query/source-scan/edge-visit counters. Transitive-closure queries, deterministic JSON sorting, alias analysis, and conflict-pair enumeration are not included in the direct-construction bound.

For the original core notation, where `T` is the actor count and `N` is the number of coalesced Arrive/LogicalWait events, this specializes to expected-amortized `O(TN)`.

## Extended SM90 protocols

TMA transaction completion, lifecycle reuse, CTA/named/cluster barriers, cp.async groups, WGMMA and TMA-store typed endpoints, channels, proxy visibility, and explicit async-engine ordering use the same finite transition framework. They are covered by focused tests and maximal-versus-memoized-all-state crosschecks over the operational FSM.

They are not consequences of the Arrive/LogicalWait theorem. [`EXTENDED_ASSURANCE.md`](EXTENDED_ASSURANCE.md) gives the separate finite-product argument for completion fairness normalization, held-anchor trace absorption, full-boundary PDG exactness, and typed visibility-aware window separation in the validated exact domain. The 65-case independent endpoint/visibility enumeration checks each extension class against both execution engines.

## Frontend obligation

A core result is only as sound as the PGIR extraction. TileLang and Triton adapters must emit complete provenance, preserve actor/participant identities, regions, generations, group endpoints, engine evidence, proxy direction, and unsupported constructs. An adapter that cannot prove a required field must return a frontend failure and must not emit a SAFE-capable PGIR.

The strict snapshot adapters expose all implemented PGIR event kinds and preserve their typed fields. Automated capture emits snapshot v2: it binds a source-IR SHA-256 and compiler profile, accounts for every executable node as classified/emitting/elided, and attaches the original IR operation, recognizer rule, source location, and activation path to every canonical event. The adapter validates this certificate before discarding capture-only evidence and constructing PGIR. Legacy snapshot v1 remains a trusted-producer compatibility boundary, not evidence that a standalone capture traversed the complete compiler IR.

The standalone TIRX and TTGIR capture tools provide executable mbarrier, cp.async, TMA load/store, ProxyFence, and WGMMA paths for both DSLs. TileLang statically interprets every statement kind exposed by the pinned TIRX runtime, resolves vector/strided views and pointer binds, and lifts explicit 128-lane warpgroup protocol actors without replacing lane memory actors. Erased tensor-descriptor shape/stride/order facts are accepted only through a source-SHA-bound descriptor certificate. Triton reconstructs region/block topology from native MLIR identities, unfolds statically resolved SCF, propagates carried SSA values and exact pointer/memdesc views, and separates pure payload operations from relevant control/address/mask evidence. Compiler-inserted async fences, TMA operations, `warp_group_dot/wait`, masked cp.async, empty groups, and hidden TMA wait modes normalize to existing PGIR events; unit/array attributes absent from Python bindings require source-SHA-bound binding evidence. The explicit launch-grid CTA count is separate from `ttg.num-ctas`; both must fit the selected profile. Different compiler spellings are frontend evidence for the same canonical PGIR event/FSM and never create DSL-specific checker protocols. Cross-DSL signature tests validate WGMMA and TMA-store normalization in addition to the core handoff/cp.async cases.

A CTA-scope proxy fence may be issued by an elected actor after shared writes from other actors only when the existing typed collective relation proves writer→arrive/release/observe(fencer) and the fence then covers the asynchronous source. Same-actor publication remains the direct case; arbitrary dependencies and control-only barriers do not authorize elected publication. This rule is DSL-neutral and is exercised by the real TileLang TMA-store lowering plus a reduced maximal/exhaustive regression.

Real production-size captures are accepted through the `maximal` checker with deterministic statistics and witness checks. Exhaustive exploration is reserved for reduced semantically equivalent fixtures—at least one per recognizer—and for the bounded protocol enumerations above; inability to run all-state exploration on a large unfolded kernel does not by itself invalidate its frontend profile.
