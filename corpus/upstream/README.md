# Locked upstream kernel corpus

This corpus archives one Python capsule for every statically discoverable TileLang or Triton kernel definition in the pinned upstream source trees. A capsule is source inventory, not automatically a supported PipeGuard program: concrete compiler lowering additionally requires a recipe with all factory, closure, signature, constexpr, target, option, and launch-grid evidence.

## Locked inputs

- TileLang: `2c31e071c626c0498b48dcdc2741dc8651b70dc1`
- TileLang TVM: `907a88c8791ccf33b9874821bc875e7abf624367`
- Triton: `61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f`
- Target profile: CUDA SM90, 32-lane warps

`sources.json` is the machine-readable lock. Runtime checkout paths are supplied explicitly and verified against these commits; they are not embedded as machine-specific absolute paths.

## Layout

```text
tilelang-2c31e071/
triton-61c02bcd/
  source-index.json
  capsules/<upstream-source-path>/<span>__<qualname>__<id>.py
  manifests/<candidate-id>.json
  recipes/<candidate-id>/<recipe-id>.json
  generated/                 # ignored, reproducible run output
```

Candidate identity and concrete specialization identity are separate. Every candidate has a capsule and manifest. An unresolved recipe records missing evidence without guessing it or dropping the source kernel.

## Regenerate source inventory

```bash
/usr/bin/python3 tools/upstream_extract.py \
  --dsl all \
  --tilelang-root /mnt/project_dft/xuwen/PipeGuard-asplos/tilelang \
  --triton-root /mnt/project_dft/xuwen/PipeGuard-asplos/triton \
  --output corpus/upstream
```

The extractor is static: it parses Python AST, resolves decorator aliases and writes dependency capsules. It does not import upstream modules, execute pytest bodies, invoke GPU kernels, run autotuning, or evaluate arbitrary Python expressions.

## Lowering and parse boundary

The corpus runner uses the pinned frontend environments. TileLang recipes are materialized with the original factory/`get_tir` path and lowered through public `tilelang.lower(...).device_mod`. Triton recipes are compiled through public `triton.compile(ASTSource(...)).asm["ttgir"]`. The raw compiler artifact is passed to the existing strict snapshot-v2 capture adapters and PGIR parser.

The corpus runner stops after PGIR parsing and records `checker_loaded=false`; checker verification is a separate phase over successfully normalized PGIR. This keeps frontend coverage failures distinct from synchronization or data-hazard verdicts.

Bulk generated TIR/TTGIR, snapshots, normalized PGIR, logs, and progress files live under `generated/` and are intentionally excluded from Git. Source indexes, capsules, manifests, and reviewed recipes are tracked.

## Pinned inventory and recipe coverage

The deterministic source inventory at these locks contains:

| DSL | Source files | Candidates | Lowerable recipes |
| --- | ---: | ---: | ---: |
| TileLang | 643 | 2,054 | 585 |
| Triton | 135 | 802 | 35 |

TileLang has 583 static profile-specialization recipes plus two reviewed recipes. Static classification rejects 18 PrimFuncs that explicitly call Blackwell `tcgen05` operations and two PrimFuncs defined inside expected-failure scopes. Source-directory names do not substitute for operation evidence: 13 dependency-closed PrimFuncs under AMD/CPU test directories are attempted under the SM90 profile, producing six additional normalized PGIR programs while unknown effects and control still fail closed.

Triton has 17 reviewed recipes and 18 statically inferred recipes. The reviewed set now grows by reading each real launch/warmup/compile context and recording one concrete recipe per statically proven parameter set; multiple parameter sets for one candidate remain separate recipes over one source capsule. The existing launch extractor remains useful as an index and rejection diagnostic, but it is no longer the primary coverage-expansion mechanism. Ordinary tensor addresses, dynamic pruning/hooks, opaque heuristics, unknown grids, aliases, and unknown compiler options remain fail-closed rather than being filled from observed runtime accidents.

An optional report-only rejection taxonomy keeps extractor limitations separate from immutable corpus evidence. Before applying reviewed overlays, automatic analysis partitions the corpus into 20 candidates with inferred recipes, 84 helpers, 180 unclassified candidates, 30 unresolved autotune roots, 441 ordinary runtime-parameter roots, 16 all-constexpr roots, and 31 zero-argument roots. Reviewed batches 1 and 2 raise the tracked source index to 34 candidates with 35 recipes without changing those automatic classifications. At relevant call sites, the largest automatic-rejection classes remain unknown runtime values, unknown grids, non-single-CTA grids, unknown autotune config construction, and dynamic autotune effects. The report lives under ignored `generated/` output and does not alter manifests, recipe IDs, or fingerprints.

The specialization rules were checked against locked Triton's native binder on 288 CUDA/HIP scalar, pointer, flag, and boundary combinations with zero mismatches. Truthy `jit_cache_hook` probes confirmed that observation returns before compilation, callable-grid evaluation, or launch. Probe output contains no pointer addresses or unredacted cache keys and is not tracked.

## AI-reviewed Triton curation batch 1

The first manually reviewed batch covers ten concrete launch specializations from `python/test/unit/language/test_line_info.py` and `python/test/unit/runtime/test_cache.py`. Pointer dtype and divisibility are recorded only for positional dtype-only warmups covered by Triton's `MockTensor` contract; launches with ordinary tensor addresses remain rejected when alignment or alias facts are not provable.

| Stage | Result |
| --- | ---: |
| Reviewed concrete recipes | 10 |
| Source/capsule native TTGIR parity | 10 |
| Strict snapshot + PGIR | 3 |
| Maximal SAFE | 3 |
| Exhaustive/maximal crosscheck | 3 matched |
| Fail-closed at capture | 7 |

The seven capture rejections are explicit: four have multiple external pointer bases without alias evidence, one noinline call lowers to two `tt.func` definitions while capture currently requires one, and two contain unsupported `tt.print`. The three captured kernels are certified empty protocol graphs; both checker engines return `SAFE`. A fresh Triton-only run over all 31 current recipes completed in 116.4 seconds with 18 `CAPTURED` and 13 `UNSUPPORTED`; an immediate repeat resumed 31/31 SHA-verified results in 0.207 seconds. Generated TTGIR, snapshots, PGIR, and run outputs remain ignored.

## AI-reviewed Triton curation batch 2

The second batch reviews the remaining statically closed launch sites around global, builtin, keyword-only constexpr, helper, and closure behavior in `python/test/unit/runtime/test_cache.py`. Four concrete recipes pass source-derived capsule replay parity, strict capture, PGIR normalization, maximal checking, and exhaustive crosscheck; all four are certified empty protocol graphs and `SAFE`.

A closure instance with `cst=tl.constexpr(42)` exposed that JSON materialization had erased the constexpr wrapper. The Triton worker now accepts only the explicit `{"__pipeguard_triton_constexpr__": value}` resolver marker and reconstructs `tl.constexpr(value)` before resolving the source-derived capsule. Malformed markers and non-scalar payloads fail closed. A separate launch was withdrawn because Triton's special global/local binding semantics were not preserved by the dependency slice; its actual source behavior is not guessed from the incomplete capsule.

A fresh Triton-only run over all 35 current recipes completed in 159.5 seconds with 22 `CAPTURED` and 13 `UNSUPPORTED`; an immediate repeat resumed 35/35 SHA-verified results in 0.209 seconds. Generated artifacts remain ignored.

## Frontend-only full run

Run the current concrete recipe set with a 150-second dual-DSL heartbeat:

```bash
/usr/bin/python3 tools/upstream_run.py \
  --dsl all \
  --tilelang-root /mnt/project_dft/xuwen/PipeGuard-asplos/tilelang \
  --triton-root /mnt/project_dft/xuwen/PipeGuard-asplos/triton \
  --triton-ptxas /usr/local/cuda-12.9/bin/ptxas \
  --output corpus/upstream/generated/final-specialization-run \
  --jobs 6 \
  --progress-interval 150 \
  --timeout 300 \
  --parse-timeout 60
```

The last full dual-DSL baseline predates reviewed batch 1: it selected 606 recipes and completed in 646.7 seconds. The current inventory contains 620 recipes; its next fresh dual-DSL run is deferred until several small reviewed batches have accumulated.

| DSL | CAPTURED | PROFILE_SKIPPED | UNSUPPORTED | LOWERING_ERROR | TIMEOUT | IMPORT_ERROR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TileLang | 95 | 51 | 146 | 292 | 1 | 0 |
| Triton | 15 | 0 | 6 | 0 | 0 | 0 |

These counts have distinct meanings:

- 292 TileLang recipes failed in the pinned compiler's native lowering or strict reference/capsule parity; compiler/transform fixtures that are not statically identifiable as expected failures remain visible here rather than being labeled from historical outcomes.
- 51 TileLang recipes were outside the selected parsing profile: 44 required multiple CTAs and seven lowered to no device kernel.
- 146 TileLang raw-IR cases reached capture and failed closed. The largest recognizer classes were unknown nested expression effects (41), unresolved static `Bind` values (24), unresolved participants (22), dynamic control (15), and unsupported barrier/address/attribute forms. Unknown dtype, alias, proxy, region, and TMA evidence also remain explicit failures.
- One TileLang native worker exceeded 300 seconds. No PGIR parser exceeded its isolated 60-second timeout.
- The six Triton failures remain explicit: two use multiple external pointer bases without SHA-bound alias evidence, and four preserve runtime scalar loop/control values as dynamic SSA under native specialization. The extractor does not rewrite those runtime values as constexpr merely because one launch supplied a literal. Scalar cases whose native TTGIR is statically closed, single-pointer accesses, and certified empty protocol graphs are captured.

In total, 309 recipes produced 310 raw device-IR units. The 110 captured recipes emitted 111 strict snapshots and 111 normalized PGIR units because one recipe contains two device kernels. `pipeguard.checker` remained unloaded throughout the runner. Repeating the command validated every fingerprint and artifact hash and resumed all 606 results in 5.92 seconds.

## Checker verification

Every normalized PGIR unit in the last full dual-DSL baseline was checked separately with the maximal engine after frontend capture:

| DSL | SAFE | DATA_HAZARD | SYNC_HAZARD | Timeout/error |
| --- | ---: | ---: | ---: | ---: |
| TileLang | 72 | 22 | 2 | 0 |
| Triton | 15 | 0 | 0 | 0 |

All 15 baseline Triton cases matched the exhaustive engine under `crosscheck`; reviewed batches 1 and 2 add seven independently crosschecked `SAFE` cases. The largest TileLang case contains 1,048,576 events: loading took 22.28 seconds and maximal checking took 42.58 seconds, producing a deterministic `DATA_HAZARD` witness. Five 262,144-event cases each loaded in about 5.1--5.9 seconds and checked in about 8.0--8.2 seconds. The memory-only fast path records one canonical transition per event and analytically equivalent singleton held-anchor boundaries; protocol-bearing fixtures continue through incremental held-anchor analysis and reduced exhaustive crosschecks.
