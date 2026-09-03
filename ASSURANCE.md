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

The strict snapshot adapters expose all implemented PGIR event kinds and preserve their typed fields. The standalone TIRX and TTGIR capture tools automate smaller, explicitly documented subsets and reject every unknown executable node. Therefore cross-DSL snapshot equivalence validates normalization and core behavior under the extraction certificate; it does not prove that an unsupported source kernel can be captured, nor that manually authored metadata is truthful.
