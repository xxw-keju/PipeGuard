# Extended SM90 assurance argument

This document states the exact domain and proof obligations for PipeGuard's finite SM90 protocol layer. It extends the raw NVIDIA-valid Arrive/LogicalWait core argument without claiming support for arbitrary GPU programs.

## Supported domain

A module is in the extended exact domain only when validation establishes all of the following:

- finite statically resolved control flow and event identities;
- exact actor, lane, CTA, resource, lifecycle, generation, group, slot, and completion identities;
- fixed positive capacities and gap-free group/engine sequences;
- exact memory allocation, address-space, offset, and extent information;
- statically complete participant sets and logical-wait identities;
- coalesced logical waits with resolved NVIDIA target generations;
- for each completed `.layout::v0` primary phase, at least one successful wait before any subsequent-phase arrive-on;
- every asynchronous operation has one finite typed terminal endpoint and satisfies its declared progress prerequisites;
- alias precision is `EXACT` and all visibility transfers have resolved direction, scope, recipients, and region coverage;
- unsupported operations, dynamic predicates/topology, unresolved aliases, and incomplete frontend extraction are rejected before `SAFE`.

The exact domain contains the implemented PGIR-v1 single-CTA protocols and the isolated PGIR-v2 single-cluster subset. It excludes grid synchronization, dynamic clusters, DSMEM TMA/proxy operations, atomics, fork/join not represented as static actors, and unknown external effects.

## Operational-risk theorem

Let `P` be a validated finite module in the extended exact domain. Its state is the finite product of actor frontiers and local protocol states. Program events and synthetic transitions are the only state transitions. A protocol-invalid attempt is observed at an enabled frontier but is not added as a valid successor. A valid stuck state is non-final and has no enabled program event or guaranteed completion/release/handoff.

The checker result satisfies:

```text
SYNC_SAFE(P)
iff
there is no reachable protocol-invalid attempt and no reachable valid stuck state.
```

### Argument

1. Every successful implementation step applies one declarative FSM rule and monotonically advances an actor, lifecycle, invocation, group endpoint, or completion identity. Failure is checked before mutation; failure-atomic tests cover every synthetic transition class.
2. Each finite trace therefore corresponds directly to an operational trace with the same stable identities. A reported protocol witness contains the valid prefix, candidate event, and pre-state, establishing report soundness.
3. Guaranteed completions are finite and inflationary. Their prerequisites are explicit: issued TMA bytes plus mbarrier arrivals, commit group, TMA-store source release, or full collective arrival. Under the declared fairness assumption, delaying a continuously enabled completion can be normalized by commuting it left across independent steps. Eager synthetic scheduling therefore does not remove a final, stuck, or invalid witness.
4. For a held anchor, any valid trace that avoids the anchor is absorbed by maximal closure unless closure first finds another real risk. Program frontiers are monotone; same-resource steps that do not commute are ordered by generation, invocation, owner/group sequence, endpoint prerequisite, or explicit engine predecessor. These are exactly the scheduler readiness predicates.
5. Running one delayed stream for every program and synthetic anchor therefore realizes every first non-commuting boundary. If a reachable invalid or stuck configuration exists, one stream absorbs its shortest valid prefix and exposes its offending frontier. Conversely, every reported frontier is operationally reachable by points 1–3.

The deterministic minimum-ready choice fixes witness selection only. It is not added to the semantic order.

## PDG theorem

For a `SAFE` module, let `MustBefore(x, y)` mean that every complete fair operational execution containing `y` contains `x` earlier. The full-boundary graph contains intrinsic program/protocol edges and, for every held anchor `x`, an edge from `x` to each unfinished software frontier at maximal quiescence.

Within the extended exact domain:

```text
PDG.reachable(x, y) iff MustBefore(x, y).
```

Intrinsic edges are operational prerequisites. A boundary edge is sound because holding `x` and exhausting every other enabled transition still cannot execute the boundary frontier `y`; releasing `x` is therefore necessary before `y`. For completeness, if no path exists, trace absorption lets the held-`x` closure execute `y` or all of its prerequisites without `x`, constructing an execution that refutes `MustBefore(x, y)`. Typed engine order contributes an edge only when explicit `engine_id/engine_seq` evidence exists.

Execution order does not imply memory visibility. Visibility is checked separately.

## Data-safety theorem

Every supported memory effect is a point or a typed interval:

```text
TMA load: issue -> mbarrier generation phase completion
cp.async: issue -> committed-group completion
WGMMA shared operands: issue -> OPERANDS_RELEASED
WGMMA accumulator: issue -> RESULT_READY
TMA store source: issue -> SOURCE_RELEASED
TMA store destination: issue -> GLOBAL_COMPLETE
```

For exact aliases, a conflicting pair is safe exactly when one direction has both:

1. PDG must-order separating the required endpoint from the other access; and
2. the required typed visibility transfer or completion observation.

The implemented transfer cases are exhaustive for the supported domain: mbarrier publication/observation, typed collective release/observe, generic-to-async ProxyFence with matching direction/scope/region/mask, and async-to-generic typed completion observation. A ProxyFence normally publishes its issuing actor's writes; an elected issuer may additionally publish another actor's shared write only when a typed collective transfer first carries that write to the issuer. Mere dependency reachability or a control-only barrier is insufficient. Engine order alone is deliberately excluded from visibility.

Therefore:

```text
checker returns SAFE
implies every may-alias conflicting pair is separated and visible.

checker returns DATA_HAZARD
iff some supported exact-alias conflicting pair lacks that separation.
```

This completeness statement depends on exact frontend effects and aliases. Unknown effects or aliases are outside the domain and fail closed rather than becoming a conservative `DATA_HAZARD` result.

## Complexity

Let:

```text
N_ext = real events + finite synthetic transitions + explicit references
T_eff = nonempty program and synthetic anchor streams
```

Each delayed stream shares one incremental state across all of its anchors. Within one stream, each transition is successfully applied at most once; each dependency, waiter, resource follower, endpoint successor, and Remaining identity generates constant expected-amortized hash/index work. Thus one stream costs expected-amortized `O(N_ext)`, and all streams cost `O(T_eff * N_ext)`.

Each anchor emits at most one frontier per software actor. Ordered asynchronous completions are represented by a linear adjacent-completion chain, and each group wait points only to its latest required completion; transitive reachability recovers all earlier completion dependencies without quadratic direct edges. The direct PDG therefore has `O(N_ext)` intrinsic edges and at most `O(T_eff * N_ext)` boundary entries, so direct construction has the same bound. Cached reachability, conflict-pair enumeration, frontend extraction, and deterministic serialization are reported separately and are not included.

For the original actor/event notation this specializes to expected-amortized `O(TN)`.

## Independent bounded validation

`tests/test_bounded_sm90_oracle.py` evaluates 113 combinations using independent endpoint/visibility formulas that do not call scheduler readiness or transition application:

- 61 cp.async, WGMMA, TMA-store, TMA ProxyFence, and TMA-load combinations;
- 16 CTA-barrier arrival/observation combinations;
- 36 ordinary-memory interval-overlap and access-mode combinations.

`tests/test_bounded_core_oracle.py` separately enumerates 460 mbarrier programs, including the successful-wait-before-first-next-arrive rule and final-phase observation obligation.

Every expected SAFE/HAZARD result is crosschecked between maximal and memoized all-state exploration and requires a witness for non-SAFE results. This bounded enumeration validates the implementation of the theorem's cases; it is not used as a substitute for the quantified argument above.
