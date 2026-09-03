# PipeGuard checker core

A small reference checker for finite, statically resolved, single-CTA SM90 PGIR programs. It is DSL-neutral and intentionally fail-closed.

## Install and run

```bash
python3.12 -m pip install -e '.[test]'
python3.12 -m pipeguard validate tests/fixtures/safe_handoff.json
python3.12 -m pipeguard normalize tests/fixtures/safe_handoff.json
python3.12 -m pipeguard check tests/fixtures/safe_handoff.json
python3.12 -m pipeguard check tests/fixtures/safe_handoff.json --engine exhaustive --pretty
python3.12 -m pipeguard check tests/fixtures/safe_handoff.json --engine crosscheck
python3.12 -m pipeguard explain tests/fixtures/safe_handoff.json
pytest
```

Every command writes deterministic compact JSON by default; pass `--pretty` for indented output. `check` and `explain` return one of `SAFE`, `SYNC_HAZARD`, `DATA_HAZARD`, `UNSUPPORTED`, `INCONCLUSIVE`, or `INTERNAL_ERROR`, and include top-level execution `statistics`.

## Reference semantics

Each actor advances in sequence order. Explicit `depends_on` references and protocol waits may additionally block a program event. Synthetic transitions are separate from program events: a `TmaLoadIssue` opens a global-source read window and a shared-destination write window, then enables one stable transaction-completion transition; only that transition credits bytes. A phase-completion transition is separately enabled when arrival count is zero and completed bytes equal expected bytes. It publishes the logical generation and advances the barrier; only then may events for the new current generation mutate barrier state. Future-generation arrivals or transaction operations are never buffered in a lookahead slot. Raw parity is valid only when it equals `logical_generation % 2`.

CTA and named barriers use a separate collective FSM. Every `BarrierArrive` covers the issuing actor's complete lane set; the resource capacity must exactly match its disjoint participants, and a CTA barrier must cover the whole CTA. A stable synthetic release is enabled only after all required lanes arrive. `BarrierObserve` blocks until that invocation releases. Release order and memory visibility are separate: a typed visibility mask grants only its declared shared/global/tensor domains, while `control_only` grants no data visibility.

The deterministic scheduler always chooses the lowest stable synthetic completion ID before the lowest `(actor_id, sequence_number, event_id)` program event. The default `maximal` engine implements that order with a hierarchical static-rank word bitmap and delta notifications rather than repeated full enabled-transition scans. Dependencies, blocked waits, future-generation phase followers, TMA spawns, and changed barrier resources explicitly notify the ready index.

Held anchors are incremental streams: one stream per actor program sequence, one naturally ordered lifecycle/generation phase stream per resource, and one singleton stream per TMA completion. Each stream starts from one fresh state. It holds its first anchor, runs to candidate-ready quiescence, records the full software-actor frontier, installs the next hold, and only then releases the old anchor. A typed `END` marker releases and flushes the final anchor. Invalid released candidates are applied and reported, while genuinely blocked quiescent states retain a stuck witness. The canonical maximal run uses the same incremental scheduler. Compact traces retain only stable step, transition, and kind fields; deterministic statistics certify stream counts, transition work, notifications, boundary evaluations, and zero full scans/clones/prefix replays.

Let `N_ext` be the number of program transitions, finite TMA completions, unique phase completions, and explicit dependency references; let `T_eff` be the number of non-empty program-actor, barrier phase-engine, and independent async-engine anchor streams. The implementation work bound is `O(T_eff * N_ext)` plus emitted boundary entries. It does not claim one shared mutable state across unrelated streams. `exhaustive` retains memoized all-state exploration as the independent small-model oracle. `crosscheck` compares complete boundaries for `SAFE` and `DATA_HAZARD`; for an early protocol result it compares only the verdict because the engines may select different deterministic witness prefixes.

The data checker constructs a full-boundary PDG with program-order, explicit dependency, arrival/release, issue/completion, and completion/wait edges. Shared visibility requires a matching actor/resource/generation arrival→phase→wait handoff; arbitrary waits do not make a path visible. It checks point read/write conflicts and both TMA-load source and destination windows. Because proxy fences are unsupported, generic writes feeding a TMA source cannot be proven safe. Unknown aliases, unresolved participant/parity facts, unknown fields/events, and unsupported protocol kinds never produce `SAFE`.

## Assurance scope

The held-anchor theorem applies to the fresh, transaction-free mbarrier core with its static actor and phase streams. Transaction completions, multi-lifecycle behavior, and collective release streams are supported by the implementation and covered by focused tests, but are stated conservatively as extensions crosschecked against the independent exhaustive oracle rather than as part of that theorem.

## Deliberate unsupported scope

PGIR v1 reserves typed resource/event fields and collision-safe synthetic identities for CTA/named barriers, commit groups, typed completion endpoints, channels, proxy visibility, and explicit async-engine ordering. This shared substrate does not by itself expand the supported execution semantics: non-mbarrier resources and the corresponding protocol events remain fail-closed until their FSM checkpoints land. Optional `engine_id`/`engine_seq` evidence on an existing TMA load is validated against a gap-free `ASYNC_ENGINE` stream, but completion ordering is not yet used to prove safety; such completions remain conservative singleton streams.

Known PGIR events outside the implemented mbarrier/TMA-load/CTA/named-barrier core (cp.async groups, TMA stores, WGMMA, channels, fork/join, atomics, and proxy fences) are recognized and reported as `UNSUPPORTED`; they are not ignored. Cluster/multi-CTA operations, dynamic participants, unresolved parity, symbolic/unknown aliasing, and non-finite control flow are also fail-closed.
