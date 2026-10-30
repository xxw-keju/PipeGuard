# PipeGuard core

This branch distributes the checker, packaged DSL capture adapters, and unit
regressions. Benchmark corpora, recipes, campaign runners, and evaluation
results are not included. The checker is based on revision `8839a7a2`.

## Optional TileLang integration

```bash
python -m pip install .
```

The `pipeguard.integration.check_tilelang_ir` API accepts a serialized lowered
TileLang device module, an exact launch grid, and optional exact scalar and TMA
descriptor evidence. Capture and checking run in a child process with a shared
120-second default deadline. Timeout is INCONCLUSIVE. The TileLang runtime is
optional; PGIR checking needs only the dependencies declared in pyproject.toml.

This is the experimental 0.2 implementation. Known limitations include collective
execution/visibility aggregation and schedule-dependent phase checks; a separate
semantic refactor is in progress. SAFE is the current model's verdict, not an
independent proof of safety of the DSL or generated CUDA. Results cover a selected
CTA and do not check cross-CTA races or numerical correctness. The optional final
PTX register checks remain disabled by default.


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

TileLang capture loads its compiler runtime only when requested. Save the post-lowering device module and capture a strict snapshot:

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
pipeguard-capture-tilelang \
  device.tvm.json --source path/to/kernel.py \
  --descriptor-evidence descriptors.json > kernel.tilelang.json
/usr/bin/python3 -m pipeguard check kernel.tilelang.json \
  --frontend tilelang --engine crosscheck
```

`--descriptor-evidence` is needed only when TMA descriptor lowering erased source shape/stride facts; non-TMA and directly addressed operations omit it.

`tilelang-snapshot-v2` captures exact single-CTA TIRX memory, full-CTA barriers, mbarrier lifecycles/transactions, cp.async, TMA load/store, ProxyFence, and WGMMA typed groups. Different lowered spellings normalize to the existing PGIR FSMs. The real warp-specialized TMA+WGMMA fixture adds explicit 128-lane `WARPGROUP` actors for collective protocol operations while retaining lane actors for ordinary memory, and records fork/handoff dependencies at the static call path. TMA descriptors whose shape/stride facts were erased from device IR require optional SHA-bound `--descriptor-evidence`; each multidimensional tile is split into exact existing `TmaLoadIssue` regions. The real elected-lane TMA-store fixture validates CTA-barrier publication into a single proxy fence.

Its static interpreter handles every TIRX statement class in the pinned compiler: sequence/attribute/buffer declarations, static branch and `For`, statically terminating `While` with `Bind` updates, assertions, `SBlock`/`SBlockRealize` iter bindings and predicates, reduction init, stores, and evaluations. There is no fixed 256-iteration semantic cap; the committed 300-iteration source→lowered-IR fixture is checked with the maximal engine. Pure expression calls are elided only when `TCallEffectKind` proves them side-effect free, nested reads are still emitted, and unknown/read-state/update-state/opaque calls fail closed without crossing the TVM callback boundary as an unstructured exception.

Ordinary regions include dtype lanes, `elem_offset`, explicit strides, view data identity, and lane-qualified local allocations. Multiple external bases require `tirx.noalias`; mbarriers require aligned eight-byte storage. `tvm_access_ptr` source/destination intervals and wait counts must be static. An operand-free `fence_proxy_async` is accepted only when the immediately following supported async issue resolves its target domain and complete source region; otherwise capture fails. Every emitted event carries its original IR operation, recognizer rule, source location, and activation path, while the snapshot binds source-IR SHA-256, compiler stage, target rule and complete node accounting. CUDA `sm_90`/`sm_90a` with 32-lane warps is accepted structurally; Blackwell, AMD, non-unit block grids, symbolic control/regions, unknown effects, unsupported address spaces, and incomplete extraction fail closed. Legacy `tilelang-snapshot-v1` remains readable only as a trusted-producer format. The committed v2 fixtures were produced by TileLang `2c31e071` and checked end to end without importing TileLang into the core environment.

## Triton frontend

Triton capture likewise runs outside the checker and uses Triton's MLIR bindings rather than parsing text with regular expressions:

```bash
pipeguard-capture-triton \
  kernel.ttgir.mlir --source path/to/kernel.py --grid-ctas 1 \
  --binding-evidence binding.json > kernel.triton.json
/usr/bin/python3 -m pipeguard check kernel.triton.json \
  --frontend triton --engine crosscheck
```

`--binding-evidence` is needed only for unit/array attributes hidden by the pinned Python binding, including `read_only` waits and warp partition geometry; its source digest must match exactly.

Shared local stores and initialized local allocations retain all physical lane/warp writers (`STORE_ALL`), including identical-value replicas. Global stores retain their separate redundant-thread predicate contract. A barrier after replicated shared stores does not remove their WAW conflict.

Final PTX WGMMA register-order checking is **disabled by default**. The default verdict covers PGIR semantics and assumes correct compiler lowering; it does not validate codegen register ordering. Supply the same-compilation PTX with `triton_capture.py --final-ptx kernel.ptx` to embed SHA-bound device-code evidence. Enable the supplementary check explicitly with `python -m pipeguard check pgir.json --check-ptx-registers`, or `python -m pipeguard check snapshot.json --frontend triton --final-ptx kernel.ptx --check-ptx-registers` for external PTX. Both `check` and `explain` support the option with PGIR, TileLang, or Triton inputs. Merely embedding or supplying PTX does not enable it. The Python API uses `check_module(module, check_ptx_registers=True)`.

When enabled, the device check independently checks WGMMA accumulator/input-A register ordering, including ordinary instructions inserted after the last register fence. `WGMMA_REGISTER_FENCE_MISSING` includes PTX line numbers and affected registers; it turns an otherwise SAFE result into SYNC_HAZARD and is retained alongside an existing DATA finding. Correct same-shape accumulator chaining needs no extra fence. Proxy fences and completion waits cannot replace the register fence. This check currently requires one kernel with straight-line WGMMA control flow; unresolved branches, calls, predicates, or instruction formatting turn an otherwise SAFE result into UNSUPPORTED. The output records `validation_scope` and `device_validation`: by default the scope is `PGIR` and `wgmma_register_fences` is `DISABLED`; if enabled without PTX, the scope remains `PGIR` and the check is `NOT_PROVIDED`. Old snapshots that discarded shared writers must be recaptured; adding PTX only checks register ordering, not arbitrary PTX memory addresses. This is not a complete PTX verifier.

Triton TMA capture keeps outstanding expected bytes and pending arrivals when assigning generations. Issuing a panel only consumes the static unissued-byte budget; it does not complete hardware transactions. Repeated `arrive.expect_tx` while arrivals are exhausted remains in the current generation and is rejected by the protocol checker. Mixed plain/TMA arrival patterns whose generation assignment is not established fail closed.

Plain Triton mbarrier arrivals retain the compiler's partition barrier before leader election. Logical generations follow the initialized arrival quorum across partitions, including multiple arrivals in one phase; ambiguous concurrent prefixes fail closed. Exact predicated `bar.sync` inline assembly ranks invocations per physical lane. Partial WGMMA aligned waits retain a control-only 128-thread participation obligation instead of becoming full-warpgroup completion events. Multiple final-PTX proxy fences are mapped only for one straight-line kernel whose committed WGMMA groups bijectively match captured dots; conditional or ambiguous mappings are rejected.

`cp.async.mbarrier.arrive.noinc` is captured as `CpAsyncMBarrierArrive`, an asynchronous arrival covering the issuing lane's preceding copies, including uncommitted copies ([PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)). It neither commits a group nor blocks the thread and does not publish generic memory accesses. Copy completion endpoints feed the deferred arrival and the target mbarrier phase; successful phase observation can order a WGMMA consumer and establish completion through an explicit return handoff. Intervening accesses are checked for conflicts rather than rejected merely for appearing after noinc. Incrementing arrivals and unresolved mixed-arrival generations remain unsupported. The production parametric engine handles these events directly; expanded reference modules use the bounded exhaustive oracle (which can reach its state limit). Copies left uncommitted and not tracked by an mbarrier are still reported at termination.

`triton-snapshot-v2` captures one post-pipeline `cuda:90` TTGIR function with static warp geometry and explicit `--grid-ctas 1` launch evidence. `ttg.num-ctas = 1` alone is only cluster geometry and is not accepted as proof of a single-CTA launch. The frontend reconstructs region/block topology from native MLIR bindings, statically unfolds `scf.for`, selected `scf.if`, `scf.execute_region`, and terminating `scf.while` with carried SSA values, and rejects dynamic or repeating control states. Static `ttg.warp_specialize` uses SHA-bound partition geometry to create disjoint actors and explicit fork/join dependencies; dependencies do not grant memory visibility. `arith.*` and `math.*` are classified as pure payload computation; a result that affects control, address, mask, or protocol evidence must still be statically evaluated at its consumer.

Ordinary global load/store, initialized shared allocations, tokenless/token-observed local loads, static masks, loop-dependent pointer offsets, disjoint pointer footprints, footprint-preserving aliases, and static `ttg.memdesc_index` views normalize to exact memory events. cp.async accepts direct or statically masked copy forms; a merged masked copy expands into existing `CpAsyncIssue` events, while empty commit and zero-token wait retain the existing group FSM. `ttng.barrier_expect` plus zero-coordinate tensor descriptors normalize to TMA-load transaction events; compiler-fenced TMA stores normalize to source/global typed groups; compiler-fenced `warp_group_dot/wait` normalizes to ProxyFence plus existing WGMMA fence/issue/commit and both completion endpoints. Python bindings hide unit/array attributes such as TMA-store `read_only`; those forms require optional SHA-bound `--binding-evidence`, with absence or mismatch failing closed rather than guessing. No Triton-specific protocol is introduced.

Multiple external bases without alias evidence, nonzero/noncontiguous tensor-descriptor coordinates without exact binding evidence, masks/views/tokens, hidden attributes without evidence, atomics, arbitrary inline assembly, dynamic control, unknown effectful operations, non-SM90 targets, and multi-CTA launches fail closed. Snapshot v2 binds source/profile/accounting and per-event recognizer provenance; v1 remains trusted-producer compatibility only. The committed fixtures are parsed by Triton `61c02bcd`, then validated by the dependency-free core process.

Equivalent frontend snapshots can be alpha-normalized and checked together:

```bash
/usr/bin/python3 -m pipeguard compare tilelang.json triton.json \
  --left-frontend tilelang --right-frontend triton
```

The semantic signature renames actor, allocation, region, resource, and event identities structurally; removes producer/source labels and DSL-specific parity derivation names; and retains topology, alias intervals, protocol generations, dependencies, completion parameters, and visibility masks. `compare` succeeds only when both the semantic digest and checker verdict/primary diagnostic agree.

Both strict snapshot adapters map every currently implemented PGIR event, including TMA load/store, cp.async commit/wait groups, WGMMA typed endpoints, ProxyFence, and explicit async-engine evidence. Reserved Channel names remain in the PGIR schema, but frontend operation mapping and semantic validation reject them as unsupported. Their fields are passed without inference to the PGIR schema and semantic validator. Standalone capture has real mbarrier, cp.async, TMA load/store, proxy-publication, and WGMMA paths for both DSLs; high-level, token-chain, split/merged, and compiler-fenced spellings all terminate in the same canonical events. Compiler-integrated producers remain necessary when a public binding erases descriptor, partition, or attribute evidence. An IR form whose actor, region, token, owner, phase, endpoint, or hidden attribute cannot be recovered directly or from SHA-bound typed evidence remains `UNSUPPORTED`, never silently omitted.

## Reference semantics

Each actor advances in sequence order. Explicit `depends_on` references and protocol waits may additionally block a program event. Synthetic transitions are separate from program events: one `TmaLoadIssue` carries one or more exact source/destination transfer records and opens their asynchronous memory windows. It credits issued bytes to its mbarrier generation; one phase-completion transition is enabled when the arrival count is zero and issued bytes equal expected bytes. That generation-level completion closes every transfer window in the phase. Legacy single-transfer records remain readable and normalize to the same canonical `transfers` form, while new captures emit only grouped TMA issues. It publishes the logical generation and advances the barrier; only then may events for the new current generation mutate barrier state. Future-generation arrivals or transaction operations are never buffered in a lookahead slot. Raw parity is valid only when it equals `logical_generation % 2`.

A `CP_ASYNC` resource keeps independent state per owner. `CpAsyncIssue` stages a global-read/shared-write operation; `CpAsyncCommitGroup` assigns the next gap-free group sequence and creates stable group and `GROUP_COMPLETE` identities. Nonempty group completions are synthetic and ordered per owner; an empty group is completed synchronously at commit, as required by its trivially-complete PTX semantics, and therefore has no completion anchor. `CpAsyncWaitGroup(N)` advances only after all but at most the `N` most recent committed groups have completed, records an observed frontier, and rejects staged-but-uncommitted operations. CP_ASYNC, WGMMA and TMA_STORE commit groups have no semantic capacity in PGIR. Commit advances a per-owner sequence; wait_group(N) observes the prefix ending at max(0, next_sequence - N). Completion and visibility remain endpoint-specific: TMA .read does not observe GLOBAL_COMPLETE. Legacy async_group capacity fields are accepted but ignored; captures omit them. The finite input graph and group endpoints are still retained, so analysis memory grows with the input; time/state budgets are analysis limits, never synchronization hazards. See the PTX [cp.async](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group), [WGMMA](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma-wait-group) and [bulk-copy](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) wait semantics.

A `WGMMA` resource requires complete aligned 128-lane warpgroup owners. `WgmmaFenceOperand` records explicit shared-operand coverage before issue; issue and gap-free commit create separate ordered `OPERANDS_RELEASED` and `RESULT_READY` endpoint streams. `WgmmaWaitGroup(N)` explicitly names which endpoint it observes, so operand reuse cannot be justified by a result-only wait and accumulator access cannot be justified by operand release. Empty WGMMA commits are fail-closed because the NVIDIA PTX contract checked for this checkpoint does not establish cp.async-style trivially-complete behavior.

A `TMA_STORE` resource stages explicit shared-source/global-destination transfers and commits them into ordered `SOURCE_RELEASED` and `GLOBAL_COMPLETE` endpoint streams. `GLOBAL_COMPLETE` has a typed prerequisite edge from the same group's source release. `TmaStoreWaitGroup(N, READ_ONLY)` observes only source release, while `GLOBAL` observes destination completion; therefore read-only wait permits shared-source reuse but cannot justify reading the global destination. The two endpoint frontiers remain distinct; empty store groups remain fail-closed.

Producer-consumer Channel resources and events are reserved but unsupported. No Channel state or visibility edge may contribute to a `SAFE` result until a real frontend mapping and its semantic contract are established.

CTA and named barriers use a separate collective FSM. Every `BarrierArrive` covers the issuing actor's complete lane set; the resource capacity must exactly match its disjoint participants, and a CTA barrier must cover the whole CTA. A stable synthetic release is enabled only after all required lanes arrive. `BarrierObserve` blocks until that invocation releases. Release order and memory visibility are separate: a typed visibility mask grants only its declared shared/global/tensor domains, while `control_only` grants no data visibility.

PGIR v2 reuses that release FSM for a conservative `cluster_barrier` subset. The target provides a resolved three-dimensional cluster shape of at most eight CTAs; every actor and DSMEM region has a checked CTA rank, and the barrier participant set covers every CTA-qualified lane exactly once. Every invocation requires one arrive and one observe from every participant. A DSMEM access requires a completed cluster synchronization both before the access (all CTAs and shared state exist) and afterward (no CTA exits while another can access its shared memory). Only a cluster barrier with the typed `dsmem` visibility bit publishes DSMEM data; CTA-local barriers, dependencies, and control-only cluster barriers do not.

The deterministic scheduler always chooses the lowest stable synthetic completion ID before the lowest `(actor_id, sequence_number, event_id)` program event. The default `maximal` engine implements that order with a hierarchical static-rank word bitmap and delta notifications rather than repeated full enabled-transition scans. Dependencies, blocked waits, future-generation phase followers, changed barrier resources, collective releases, changed commit-group owner streams and retired explicit-engine predecessors notify only their indexed followers.

Held anchors are incremental streams: one stream per actor program sequence, one naturally ordered lifecycle/generation phase stream per resource, ordered typed group-completion streams per `(resource, owner, endpoint)`, and explicit async-engine terminal-completion streams. Every TMA load in one mbarrier generation shares that generation's phase-completion anchor; grouped protocol endpoints without engine evidence retain only their intrinsic ordering. Each stream starts from one fresh state. It holds its first anchor, runs to candidate-ready quiescence, records the full software-actor frontier, installs the next hold, and only then releases the old anchor. A typed `END` marker releases and flushes the final anchor. Invalid released candidates are applied and reported, while genuinely blocked quiescent states retain a stuck witness. The canonical maximal run uses the same incremental scheduler. Compact traces retain only stable step, transition, and kind fields; deterministic statistics certify stream counts, transition work, notifications, boundary evaluations, and zero full scans/clones/prefix replays.

Let `N_ext` be the number of program transitions, finite group completions and collective/phase releases, plus explicit dependency references; let `T_eff` be the number of non-empty program-actor, barrier phase/release, commit-group, and independent async-engine anchor streams. Scheduling and direct full-boundary PDG construction take expected-amortized `O(T_eff * N_ext)` plus emitted graph entries; transitive-closure queries, deterministic serialization, alias analysis, and conflict-pair enumeration are outside that bound. The certificate reports scheduler and PDG work separately. `exhaustive` retains memoized all-state exploration over the same operational FSM, while the NVIDIA-valid core additionally has an independent declarative bounded oracle. `crosscheck` compares complete boundaries for `SAFE` and `DATA_HAZARD`; for an early protocol result it compares only the verdict because the engines may select different deterministic witness prefixes.

Exhaustive exploration is a bounded protocol oracle, not a release requirement for fully unfolded production kernels. Every real captured fixture runs `maximal` and checks deterministic output, statistics, and witnesses. Each frontend recognizer additionally has at least one reduced, semantically equivalent fixture that runs `crosscheck`; bounded core and endpoint/visibility enumerations continue to cover protocol interleavings. A large real kernel is therefore not rejected merely because all-state exploration would exceed its practical bound.

The data checker constructs a full-boundary PDG with program-order, explicit dependency, arrival/release, issue/group/completion, completion/wait, and explicit `ENGINE_ORDER` edges. Shared visibility requires a matching actor/resource/generation arrival→phase→wait handoff; arbitrary waits do not make a path visible. It checks point read/write conflicts, every transfer region in generation-bounded TMA-load windows, cp.async source and destination windows, TMA-store shared-source and global-destination windows, WGMMA shared-operand windows through operand release, and WGMMA register-accumulator windows through result readiness. A group wait directly closes a window only for its owner; cross-actor publication is never inferred from `depends_on`. Engine order constrains terminal completion readiness but grants no memory visibility by itself; a generic consumer must still observe a typed completion. A `ProxyFence` may publish a generic write to an async read only for the supported CTA-scope `GENERIC → ASYNC_COPY/TMA/TENSOR_CORE` direction, when it is issued by the writer, lies between the write and issue in the PDG, fully covers the async source region, and carries the matching typed visibility mask. Non-bulk cp.async reads its source in the generic proxy, so an ordinary happens-before path orders an earlier write to that source without a fence. The reverse async-to-generic direction is never inferred from a symmetric fence; it requires observation of the corresponding typed completion. Unknown aliases, unresolved participant/parity facts, unknown fields/events, and unsupported protocol kinds never produce `SAFE`.

## Assurance scope

See [`ASSURANCE.md`](ASSURANCE.md) for the core theorem and 460-program raw NVIDIA-validity enumeration, and [`EXTENDED_ASSURANCE.md`](EXTENDED_ASSURANCE.md) for the separate finite SM90 protocol, PDG, typed-window, and `O(T_eff * N_ext)` argument. The additional bounded suite checks 113 typed endpoint/visibility, CTA-barrier, and ordinary-memory combinations. The held-anchor theorem applies to the fresh, transaction-free `.layout::v0` mbarrier core with its static actor and primary-phase streams, including the requirement that a successful wait precede the first arrive-on in the subsequent phase. Generation-level TMA phase completions, multi-lifecycle behavior, CTA/cluster collective releases, cp.async groups, WGMMA/TMA-store typed endpoints and explicit async-engine ordering are supported by the implementation and covered by focused tests, but are stated conservatively as extensions crosschecked against memoized all-state exploration of the same operational FSM rather than as part of that theorem.

## Deliberate unsupported scope

PGIR v1 and v2 implement optional `engine_id`/`engine_seq` evidence on TMA-load, cp.async, WGMMA, and TMA-store issue events. Each gap-free `ASYNC_ENGINE` issue stream is lifted to typed terminal endpoints (`MBARRIER_PHASE_COMPLETE`, `GROUP_COMPLETE`, `RESULT_READY`, or `GLOBAL_COMPLETE`); TMA-load issues sharing one mbarrier lifecycle/generation collapse to its single phase endpoint, while issues sharing one committed group collapse to that group's endpoint. An unresolved endpoint, one endpoint assigned to multiple engines, or a non-contiguous endpoint reappearance is fail-closed. Only this explicit evidence creates cross-operation completion order. Without it, TMA loads use their naturally ordered per-resource phase stream and group endpoints retain only intrinsic per-owner ordering.

Known PGIR events outside the implemented mbarrier/TMA-load/TMA-store/CTA/named/cluster-barrier/cp.async/WGMMA/proxy-visibility core (including channels, fork/join, and atomics) are recognized and reported as `UNSUPPORTED`; they are not ignored. Unsupported proxy directions or scopes, multi-cluster/grid synchronization, nonportable clusters larger than eight CTAs, dynamic cluster topology or participants, DSMEM proxy/TMA operations, unresolved parity, symbolic/unknown remote aliasing, and non-finite control flow are also fail-closed.
