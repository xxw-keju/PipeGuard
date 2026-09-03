# PipeGuard checker core

A small reference checker for finite, statically resolved SM90 PGIR programs. PGIR v1 is single-CTA; PGIR v2 adds an isolated static single-cluster subset. The checker is DSL-neutral and intentionally fail-closed.

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

## TileLang frontend

TileLang capture runs in its isolated environment and never imports the checker. Save the post-lowering device module and capture a strict snapshot:

```python
artifact = tilelang.lower(
    prim_func,
    target="cuda",
    enable_host_codegen=False,
    enable_device_compile=False,
)
Path("device.tvm.json").write_text(tvm.ir.save_json(artifact.device_mod))
```

```bash
/usr/local/miniconda3/envs/tl-pg/bin/python tools/tilelang_capture.py \
  device.tvm.json --source path/to/kernel.py > kernel.tilelang.json
/usr/bin/python3 -m pipeguard check kernel.tilelang.json \
  --frontend tilelang --engine crosscheck
```

`tilelang-snapshot-v1` automatically captures exact single-CTA TIRX memory accesses, full-CTA shared barriers, fresh-lifecycle mbarrier streams, and cp.async issue/commit/wait pipelines. `tvm_access_ptr` source/destination intervals and wait counts must be static. An operand-free `fence_proxy_async` is accepted only when the immediately following supported async issue resolves its target domain and complete source region; otherwise capture fails. Static lane extents, branches, loops, addresses, barrier identities, and parity expressions are mandatory. Non-unit block grids, symbolic control/regions, unknown statements or calls, unsupported address spaces, and incomplete extraction fail closed. The committed mbarrier and ProxyFence+cp.async device-IR fixtures were produced by TileLang `2c31e071` and checked end to end without importing TileLang into the core environment.

## Triton frontend

Triton capture likewise runs outside the checker and uses Triton's MLIR bindings rather than parsing text with regular expressions:

```bash
/usr/local/miniconda3/envs/triton-pg/bin/python tools/triton_capture.py \
  kernel.ttgir.mlir --source path/to/kernel.py > kernel.triton.json
/usr/bin/python3 -m pipeguard check kernel.triton.json \
  --frontend triton --engine crosscheck
```

`triton-snapshot-v1` automatically captures one post-pipeline `cuda:90` TTGIR function with one CTA, static warp geometry, one straight-line executable block, direct mbarrier SSA identity, and cp.async copy/commit/wait token chains. Async global sources must have exact contiguous provenance `scalar pointer → splat → static make_range → addptr`; destinations must be direct static `ttg.local_alloc` memdescs, commit operands must equal staged copy tokens, waits must name committed tokens with static `num`, and a following `ttg.local_load` must consume the wait token. Multiple external pointer bases without alias evidence, multiple blocks/regions, SSA aliases, dynamic operands, unknown operations, non-SM90 targets, and multi-CTA modules fail closed. The committed mbarrier and cp.async TTGIR fixtures are parsed by Triton `61c02bcd`, then validated and crosschecked by the dependency-free core process.

Equivalent frontend snapshots can be alpha-normalized and checked together:

```bash
/usr/bin/python3 -m pipeguard compare tilelang.json triton.json \
  --left-frontend tilelang --right-frontend triton
```

The semantic signature renames actor, allocation, region, resource, and event identities structurally; removes producer/source labels and DSL-specific parity derivation names; and retains topology, alias intervals, protocol generations, dependencies, completion parameters, and visibility masks. `compare` succeeds only when both the semantic digest and checker verdict/primary diagnostic agree.

Both strict snapshot adapters map every currently implemented PGIR event, including TMA load/store, cp.async commit/wait groups, WGMMA typed endpoints, channels, ProxyFence, and explicit async-engine evidence. Their fields are passed without inference to the PGIR schema and semantic validator. The standalone capture tools now provide real mbarrier and cp.async end-to-end paths for both DSLs, including TileLang ProxyFence and Triton SSA token/pointer provenance. Other implemented PGIR protocols can be emitted by compiler-integrated strict snapshot producers when their typed fields are complete; an IR form whose actor, region, token, owner, phase, or endpoint evidence cannot be recovered remains `UNSUPPORTED`, never silently omitted.

## Reference semantics

Each actor advances in sequence order. Explicit `depends_on` references and protocol waits may additionally block a program event. Synthetic transitions are separate from program events: a `TmaLoadIssue` opens a global-source read window and a shared-destination write window, then enables one stable transaction-completion transition; only that transition credits bytes. A phase-completion transition is separately enabled when arrival count is zero and completed bytes equal expected bytes. It publishes the logical generation and advances the barrier; only then may events for the new current generation mutate barrier state. Future-generation arrivals or transaction operations are never buffered in a lookahead slot. Raw parity is valid only when it equals `logical_generation % 2`.

A `CP_ASYNC` resource keeps independent state per owner. `CpAsyncIssue` stages a global-read/shared-write operation; `CpAsyncCommitGroup` assigns the next gap-free group sequence and creates stable group and `GROUP_COMPLETE` identities. Nonempty group completions are synthetic and ordered per owner; an empty group is completed synchronously at commit, as required by its trivially-complete PTX semantics, and therefore has no completion anchor. `CpAsyncWaitGroup(N)` advances only after all but at most the `N` most recent committed groups have completed, records an observed frontier, and rejects staged-but-uncommitted operations. At most eight uncompleted groups are supported; the declared resource capacity may conservatively be smaller, and held-completion exploration detects capacity overflow schedules.

A `WGMMA` resource requires complete aligned 128-lane warpgroup owners. `WgmmaFenceOperand` records explicit shared-operand coverage before issue; issue and gap-free commit create separate ordered `OPERANDS_RELEASED` and `RESULT_READY` endpoint streams. `WgmmaWaitGroup(N)` explicitly names which endpoint it observes, so operand reuse cannot be justified by a result-only wait and accumulator access cannot be justified by operand release. Capacity counts a group until every endpoint completes. Empty WGMMA commits are fail-closed because the NVIDIA PTX contract checked for this checkpoint does not establish cp.async-style trivially-complete behavior.

A `TMA_STORE` resource stages explicit shared-source/global-destination transfers and commits them into ordered `SOURCE_RELEASED` and `GLOBAL_COMPLETE` endpoint streams. `GLOBAL_COMPLETE` has a typed prerequisite edge from the same group's source release. `TmaStoreWaitGroup(N, READ_ONLY)` observes only source release, while `GLOBAL` observes destination completion; therefore read-only wait permits shared-source reuse but cannot justify reading the global destination. Capacity retains a group until both endpoints finish, and empty store groups remain fail-closed.

A producer-consumer channel owns a fixed number of physical slots with explicit logical phases. `ChannelCreate` initializes producer-available slots; acquire/commit records producer ownership, commit enables an independent stable handoff, consumer wait observes that publication, and release advances the slot phase. A next-phase acquire blocks until the prior consumer release. Typed channel visibility follows producer commit→handoff→consumer wait and consumer release→next producer acquire; `control_only` channels order execution but never discharge a data conflict.

CTA and named barriers use a separate collective FSM. Every `BarrierArrive` covers the issuing actor's complete lane set; the resource capacity must exactly match its disjoint participants, and a CTA barrier must cover the whole CTA. A stable synthetic release is enabled only after all required lanes arrive. `BarrierObserve` blocks until that invocation releases. Release order and memory visibility are separate: a typed visibility mask grants only its declared shared/global/tensor domains, while `control_only` grants no data visibility.

PGIR v2 reuses that release FSM for a conservative `cluster_barrier` subset. The target provides a resolved three-dimensional cluster shape of at most eight CTAs; every actor and DSMEM region has a checked CTA rank, and the barrier participant set covers every CTA-qualified lane exactly once. Every invocation requires one arrive and one observe from every participant. A DSMEM access requires a completed cluster synchronization both before the access (all CTAs and shared state exist) and afterward (no CTA exits while another can access its shared memory). Only a cluster barrier with the typed `dsmem` visibility bit publishes DSMEM data; CTA-local barriers, dependencies, and control-only cluster barriers do not.

The deterministic scheduler always chooses the lowest stable synthetic completion ID before the lowest `(actor_id, sequence_number, event_id)` program event. The default `maximal` engine implements that order with a hierarchical static-rank word bitmap and delta notifications rather than repeated full enabled-transition scans. Dependencies, blocked waits, future-generation phase followers, TMA spawns, changed barrier resources, collective releases, changed commit-group owner streams, changed channel slots, and retired explicit-engine predecessors notify only their indexed followers.

Held anchors are incremental streams: one stream per actor program sequence, one naturally ordered lifecycle/generation phase stream per resource, ordered typed group-completion streams per `(resource, owner, endpoint)`, one channel handoff stream per `(resource, slot)`, and explicit async-engine terminal-completion streams. TMA-load completions without engine evidence remain singleton streams; group endpoints without evidence retain only their intrinsic per-owner ordering. Each stream starts from one fresh state. It holds its first anchor, runs to candidate-ready quiescence, records the full software-actor frontier, installs the next hold, and only then releases the old anchor. A typed `END` marker releases and flushes the final anchor. Invalid released candidates are applied and reported, while genuinely blocked quiescent states retain a stuck witness. The canonical maximal run uses the same incremental scheduler. Compact traces retain only stable step, transition, and kind fields; deterministic statistics certify stream counts, transition work, notifications, boundary evaluations, and zero full scans/clones/prefix replays.

Let `N_ext` be the number of program transitions, finite TMA/group completions and collective/phase releases, plus explicit dependency references; let `T_eff` be the number of non-empty program-actor, barrier phase/release, commit-group, channel-slot, and independent async-engine anchor streams. Scheduling and direct full-boundary PDG construction take expected-amortized `O(T_eff * N_ext)` plus emitted graph entries; transitive-closure queries, deterministic serialization, alias analysis, and conflict-pair enumeration are outside that bound. The certificate reports scheduler and PDG work separately. `exhaustive` retains memoized all-state exploration over the same operational FSM, while the NVIDIA-valid core additionally has an independent declarative bounded oracle. `crosscheck` compares complete boundaries for `SAFE` and `DATA_HAZARD`; for an early protocol result it compares only the verdict because the engines may select different deterministic witness prefixes.

The data checker constructs a full-boundary PDG with program-order, explicit dependency, arrival/release, issue/group/completion, completion/wait, and explicit `ENGINE_ORDER` edges. Shared visibility requires a matching actor/resource/generation arrival→phase→wait handoff; arbitrary waits do not make a path visible. It checks point read/write conflicts, TMA-load/cp.async source and destination windows, TMA-store shared-source and global-destination windows, WGMMA shared-operand windows through operand release, and WGMMA register-accumulator windows through result readiness. A group wait directly closes a window only for its owner; cross-actor publication is never inferred from `depends_on`. Engine order constrains terminal completion readiness but grants no memory visibility by itself; a generic consumer must still observe a typed completion. A `ProxyFence` may publish a generic write to an async read only for the supported CTA-scope `GENERIC → ASYNC_COPY/TMA/TENSOR_CORE` direction, when it is issued by the writer, lies between the write and issue in the PDG, fully covers the async source region, and carries the matching typed visibility mask. The reverse async-to-generic direction is never inferred from a symmetric fence; it requires observation of the corresponding typed completion. Unknown aliases, unresolved participant/parity facts, unknown fields/events, and unsupported protocol kinds never produce `SAFE`.

## Assurance scope

See [`ASSURANCE.md`](ASSURANCE.md) for the core theorem and 460-program raw NVIDIA-validity enumeration, and [`EXTENDED_ASSURANCE.md`](EXTENDED_ASSURANCE.md) for the separate finite SM90 protocol, PDG, typed-window, and `O(T_eff * N_ext)` argument plus its 65-case independent endpoint/visibility enumeration. The held-anchor theorem applies to the fresh, transaction-free mbarrier core with its static actor and phase streams. Transaction completions, multi-lifecycle behavior, CTA/cluster collective releases, cp.async groups, WGMMA/TMA-store typed endpoints, channel handoff streams, and explicit async-engine ordering are supported by the implementation and covered by focused tests, but are stated conservatively as extensions crosschecked against memoized all-state exploration of the same operational FSM rather than as part of that theorem.

## Deliberate unsupported scope

PGIR v1 and v2 implement optional `engine_id`/`engine_seq` evidence on TMA-load, cp.async, WGMMA, and TMA-store issue events. Each gap-free `ASYNC_ENGINE` issue stream is lifted to typed terminal endpoints (`TMA_COMPLETION`, `GROUP_COMPLETE`, `RESULT_READY`, or `GLOBAL_COMPLETE`); consecutive issues sharing one committed group collapse to one endpoint. An unresolved endpoint, one endpoint assigned to multiple engines, or a non-contiguous endpoint reappearance is fail-closed. Only this explicit evidence creates cross-operation completion order. Without it, TMA completions remain conservative singleton streams and group endpoints retain only intrinsic per-owner ordering.

Known PGIR events outside the implemented mbarrier/TMA-load/TMA-store/CTA/named/cluster-barrier/cp.async/WGMMA/channel/proxy-visibility core (including fork/join and atomics) are recognized and reported as `UNSUPPORTED`; they are not ignored. Unsupported proxy directions or scopes, multi-cluster/grid synchronization, nonportable clusters larger than eight CTAs, dynamic cluster topology or participants, DSMEM proxy/TMA operations, unresolved parity, symbolic/unknown remote aliasing, and non-finite control flow are also fail-closed.
