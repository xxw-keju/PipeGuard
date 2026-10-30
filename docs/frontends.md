# Frontends and capture contracts

Capture turns a fixed compiler IR and exact launch bindings into a DSL-neutral
PGIR module. TileLang and Triton runtimes are optional and loaded on demand.
Both frontends feed the same maximal-only checker.

## TileLang

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
  --frontend tilelang
```

`--descriptor-evidence` is needed only when TMA descriptor lowering erased source shape/stride facts; non-TMA and directly addressed operations omit it.

`tilelang-snapshot-v2` captures exact single-CTA TIRX memory, full-CTA barriers, mbarrier lifecycles/transactions, cp.async, TMA load/store, ProxyFence, and WGMMA typed groups. Different lowered spellings normalize to the existing PGIR FSMs. The real warp-specialized TMA+WGMMA fixture adds explicit 128-lane `WARPGROUP` actors for collective protocol operations while retaining lane actors for ordinary memory, and records fork/handoff dependencies at the static call path. TMA descriptors whose shape/stride facts were erased from device IR require optional SHA-bound `--descriptor-evidence`; each multidimensional tile is split into exact existing `TmaLoadIssue` regions. The real elected-lane TMA-store fixture validates CTA-barrier publication into a single proxy fence.

Its static interpreter handles every TIRX statement class in the pinned compiler: sequence/attribute/buffer declarations, static branch and `For`, statically terminating `While` with `Bind` updates, assertions, `SBlock`/`SBlockRealize` iter bindings and predicates, reduction init, stores, and evaluations. There is no fixed 256-iteration semantic cap; the committed 300-iteration source→lowered-IR fixture is checked with the maximal engine. Pure expression calls are elided only when `TCallEffectKind` proves them side-effect free, nested reads are still emitted, and unknown/read-state/update-state/opaque calls fail closed without crossing the TVM callback boundary as an unstructured exception.

Ordinary regions include dtype lanes, `elem_offset`, explicit strides, view data identity, and lane-qualified local allocations. Multiple external bases require `tirx.noalias`; mbarriers require aligned eight-byte storage. `tvm_access_ptr` source/destination intervals and wait counts must be static. An operand-free `fence_proxy_async` is accepted only when the immediately following supported async issue resolves its target domain and complete source region; otherwise capture fails. Every emitted event carries its original IR operation, recognizer rule, source location, and activation path, while the snapshot binds source-IR SHA-256, compiler stage, target rule and complete node accounting. CUDA `sm_90`/`sm_90a` with 32-lane warps is accepted structurally; Blackwell, AMD, non-unit block grids, symbolic control/regions, unknown effects, unsupported address spaces, and incomplete extraction fail closed. Legacy `tilelang-snapshot-v1` remains readable only as a trusted-producer format. The committed v2 fixtures were produced by TileLang `2c31e071` and checked end to end without importing TileLang into the core environment.

## Triton

Triton capture likewise runs outside the checker and uses Triton's MLIR bindings rather than parsing text with regular expressions:

```bash
pipeguard-capture-triton \
  kernel.ttgir.mlir --source path/to/kernel.py --grid-ctas 1 \
  --binding-evidence binding.json > kernel.triton.json
/usr/bin/python3 -m pipeguard check kernel.triton.json \
  --frontend triton
```

`--binding-evidence` is needed only for unit/array attributes hidden by the pinned Python binding, including `read_only` waits and warp partition geometry; its source digest must match exactly.

Shared local stores and initialized local allocations retain all physical lane/warp writers (`STORE_ALL`), including identical-value replicas. Global stores retain their separate redundant-thread predicate contract. A barrier after replicated shared stores does not remove their WAW conflict.

Final PTX WGMMA register-order checking is **disabled by default**. The default verdict covers PGIR semantics and assumes correct compiler lowering; it does not validate codegen register ordering. Supply the same-compilation PTX with `triton_capture.py --final-ptx kernel.ptx` to embed SHA-bound device-code evidence. Enable the supplementary check explicitly with `python -m pipeguard check pgir.json --check-ptx-registers`, or `python -m pipeguard check snapshot.json --frontend triton --final-ptx kernel.ptx --check-ptx-registers` for external PTX. Both `check` and `explain` support the option with PGIR, TileLang, or Triton inputs. Merely embedding or supplying PTX does not enable it. The Python API uses `check_module(module, check_ptx_registers=True)`.

When enabled, the device check independently checks WGMMA accumulator/input-A register ordering, including ordinary instructions inserted after the last register fence. `WGMMA_REGISTER_FENCE_MISSING` includes PTX line numbers and affected registers; it turns an otherwise SAFE result into SYNC_HAZARD and is retained alongside an existing DATA finding. Correct same-shape accumulator chaining needs no extra fence. Proxy fences and completion waits cannot replace the register fence. This check currently requires one kernel with straight-line WGMMA control flow; unresolved branches, calls, predicates, or instruction formatting turn an otherwise SAFE result into UNSUPPORTED. The output records `validation_scope` and `device_validation`: by default the scope is `PGIR` and `wgmma_register_fences` is `DISABLED`; if enabled without PTX, the scope remains `PGIR` and the check is `NOT_PROVIDED`. Old snapshots that discarded shared writers must be recaptured; adding PTX only checks register ordering, not arbitrary PTX memory addresses. This is not a complete PTX verifier.

Triton TMA capture keeps outstanding expected bytes and pending arrivals when assigning generations. Issuing a panel only consumes the static unissued-byte budget; it does not complete hardware transactions. Repeated `arrive.expect_tx` while arrivals are exhausted remains in the current generation and is rejected by the protocol checker. Mixed plain/TMA arrival patterns whose generation assignment is not established fail closed.

Plain Triton mbarrier arrivals retain the compiler's partition barrier before leader election. Logical generations follow the initialized arrival quorum across partitions, including multiple arrivals in one phase; ambiguous concurrent prefixes fail closed. Exact predicated `bar.sync` inline assembly ranks invocations per physical lane. Partial WGMMA aligned waits retain a control-only 128-thread participation obligation instead of becoming full-warpgroup completion events. Multiple final-PTX proxy fences are mapped only for one straight-line kernel whose committed WGMMA groups bijectively match captured dots; conditional or ambiguous mappings are rejected.

`cp.async.mbarrier.arrive.noinc` is captured as `CpAsyncMBarrierArrive`, an asynchronous arrival covering the issuing lane's preceding copies, including uncommitted copies ([PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)). It neither commits a group nor blocks the thread and does not publish generic memory accesses. Copy completion endpoints feed the deferred arrival and the target mbarrier phase; successful phase observation can order a WGMMA consumer and establish completion through an explicit return handoff. Intervening accesses are checked for conflicts rather than rejected merely for appearing after noinc. Incrementing arrivals and unresolved mixed-arrival generations remain unsupported. Both compact and expanded modules handle these events through maximal; expanded copy completions and deferred arrivals use indexed readiness notifications. Copies left uncommitted and not tracked by an mbarrier are still reported at termination.

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


## Mapping to checker operations

| IR construct | Canonical model | Required capture evidence |
|---|---|---|
| Ordinary load/store/atomic | Memory effects and exact byte intervals | Allocation identity, offsets, widths, lane ownership, predicates |
| CTA or named barrier | BarrierArrive / BarrierObserve | Actual participants, count, invocation, visibility mask |
| mbarrier operations | Init, arrive, expect, parity wait, invalidate | Barrier identity, lifecycle, logical generation, arrival/byte count |
| cp.async | Issue, commit group, wait group | Issuer, source/destination regions, group sequence |
| cp.async.mbarrier.noinc | Copy completion followed by deferred arrival | Issuer's prior copies and destination mbarrier phase |
| TMA load | TmaLoadIssue and mbarrier completion | Exact descriptor shape/strides, coordinates, transferred bytes |
| TMA store | Issue, commit, source/global completion waits | Shared source, global destination, read-only versus full wait |
| WGMMA | Operand fence, issue, commit, typed waits | Warpgroup, operand footprints, accumulator, group ownership |
| Proxy fence | Typed visibility operation | Direction, scope, address space and covered regions |

These are the current implementation's mappings, not an assertion that all DSL
operators are thin PTX wrappers. Unsupported or erased evidence produces an
explicit diagnostic. See [semantics](semantics.md) for each protocol's behavior
and the known limitations of this release.

## TileLang convenience interface

The TileLang fork provides `tilelang.contrib.pipeguard.check(prim_func, ...)`
and `kernel.check_pipeguard(...)`. It calls the packaged
`pipeguard.integration.check_tilelang_ir` API. Inputs include an exact launch grid,
a selected CTA, scalar values, and optional pointer-table/constant-buffer evidence.
Descriptor recovery reads the compiled host IR; supplied descriptors may supplement
but must not contradict it. Evidence is SHA256-bound to the serialized device IR.

Capture and checking share a 120-second default subprocess deadline. Lowering
and serialization happen before that deadline. A timeout returns INCONCLUSIVE;
missing input evidence returns UNSUPPORTED. No GPU kernel is launched. A cached
kernel without retained IR must be checked through its original PrimFunc.
