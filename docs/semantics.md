# Checker semantics and result scope

This documents the existing experimental 0.2 implementation, based on the
previous core checkpoint. Selecting maximal alone does not implement the separate
semantic redesign. In particular, SAFE is a model result, not an independent
proof of correctness of the DSL, compiler output, or numerical result.

## Inputs and verdicts

PGIR describes finite, statically resolved events, program actors, allocations,
protocol resources, and exact launch evidence. The DSL frontends analyze a
selected CTA. Explicit PGIR v2 additionally retains a restricted static cluster
profile; the TileLang convenience API does not claim grid/cluster coverage.

| Verdict | Meaning |
|---|---|
| SAFE | No hazard found by the current model |
| DATA_HAZARD | Unordered conflicting modeled accesses, with a witness |
| SYNC_HAZARD | A modeled protocol violation or blocked completion |
| UNSUPPORTED | A construct or required input fact cannot be represented |
| INCONCLUSIVE | Analysis could not conclude, including integration timeout |
| INTERNAL_ERROR | Capture/checking infrastructure failed |

Unknown aliases, participants, parity, or operation effects are not silently
ignored. Input errors are not SAFE. PTX register checking is optional and disabled
by default (`--check-ptx-registers`); supplying PTX alone does not enable it.

## Ordinary accesses and barriers

Memory effects use allocation identity and byte intervals. An overlap with a
writer requires supported ordering/visibility evidence. Same-value writes are
not automatically exempt. Atomics are handled only for recognized operations and
supported scope/address semantics; other forms fail closed.

CTA/named barriers have explicit participants, arrival counts, invocation IDs,
and typed visibility masks. Observe waits for release. A control-only mask does
not publish memory. The static cluster subset requires CTA-qualified actors and
DSMEM regions, checked topology, and cluster synchronization around DSMEM use;
unresolved topology, multicast, or remote aliasing is unsupported.

## mbarrier and TMA loads

An mbarrier carries lifecycle, generation, pending arrivals, expected transaction
bytes, and successful-observation state. `arrive.expect_tx(n)` registers expected
bytes and performs an arrival; it does not mean those bytes have completed.
TMA issue opens an asynchronous transfer window. Consumers require a modeled
completion observation. Capture supplies generation assertions; raw parity must
match the declared logical generation modulo two.

The current representations differ internally. The expanded model uses one
synthetic phase completion after arrivals and issued/expected-byte accounting;
the compact model tracks transaction-completion actions. These mechanisms have
known phase-assignment/aggregation limitations. Neither should be described as
an exact general PTX tx state machine. The separate planned P-domain redesign is
not part of this release.

The implementation retains phase-level observation before reuse; it does not
require every participating thread to wait. Lifecycle and phase violations carry
protocol diagnostics. See the limitations below for visibility aggregation.

## Asynchronous groups

| Protocol | Issue effects | Completion observations |
|---|---|---|
| cp.async | Global read, shared write | wait_group covers a committed prefix |
| WGMMA | Shared operands, register accumulator | Separate operand-release and result-ready endpoints |
| TMA store | Shared read, global write | `.read` observes source release only; full wait observes global completion |

Group sequence is per owner. `wait_group(N)` covers the prefix ending at
`max(0, next_sequence - N)`. Commit-group count is not a semantic capacity limit.
Legacy async-group `capacity` fields are ignored and removed by normalization.
Barrier capacity is a separate arrival-count concept and remains meaningful.

The `.noinc` cp.async/mbarrier path delivers an asynchronous arrival after the
issuer's tracked copies complete. It does not commit a copy group or release the
issuer's generic memory history. In expanded PGIR, independent copy-completion
and deferred-arrival anchors now participate directly in incremental maximal
scheduling; there is no alternate-engine fallback. Other deferred-arrival forms
remain subject to capture's supported-operation checks.

Source reuse after a TMA `.read` wait does not establish visibility of the global
destination. WGMMA operand and result waits are also kept distinct. Engine-order
metadata constrains completion order; it does not independently publish memory.

## Proxy visibility

Proxy fences require a supported direction, scope, covered region and visibility
mask. Async completion observation and generic-to-async publication are distinct.
Non-bulk cp.async source reads use the generic proxy; TMA/WGMMA operands require
the corresponding modeled proxy visibility. Unknown fence forms are unsupported.

## Known limitations

The current core has demonstrated gaps involving collective rendezvous,
completion visibility, elected waiters, and schedule-dependent phase checks.
Compact parameterized execution uses a deterministic direct schedule; it should
not be described as providing the full held-anchor theorem for every protocol.
The expanded path uses held-anchor maximal scheduling. Sharing a verdict between
representations is not independent proof of correctness.

Results do not cover arbitrary runtime inputs, cross-CTA races, numerical
correctness, or changes introduced by compiler passes after capture. Accurate
bindings and a faithful frontend mapping are essential. The installed entrypoints
expose only maximal; unsupported engine names are rejected rather than silently
mapped to maximal.
