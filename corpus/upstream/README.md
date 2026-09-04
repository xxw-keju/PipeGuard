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
| Triton | 135 | 802 | 14 |

TileLang has 583 static profile-specialization recipes plus two reviewed recipes. Static classification rejects 18 PrimFuncs that explicitly call Blackwell `tcgen05` operations and two PrimFuncs defined inside expected-failure scopes. Source-directory names do not substitute for operation evidence: 13 dependency-closed PrimFuncs under AMD/CPU test directories are attempted under the SM90 profile, producing six additional normalized PGIR programs while unknown effects and control still fail closed.

Triton has three reviewed recipes, three recipes inferred from exact literal `ASTSource` plus `GPUTarget("cuda", 90, 32)` compile sites, and eight profile-derived all-constexpr single-CTA launch recipes. Dynamic grids, runtime pointer specialization attributes, autotune winners, aliases, and unknown compiler options are not guessed.

## Frontend-only full run

Run the current concrete recipe set with a 150-second dual-DSL heartbeat:

```bash
/usr/bin/python3 tools/upstream_run.py \
  --dsl all \
  --tilelang-root /mnt/project_dft/xuwen/PipeGuard-asplos/tilelang \
  --triton-root /mnt/project_dft/xuwen/PipeGuard-asplos/triton \
  --triton-ptxas /usr/local/cuda-12.9/bin/ptxas \
  --output corpus/upstream/generated/final-current-all \
  --jobs 6 \
  --progress-interval 150 \
  --timeout 300 \
  --parse-timeout 60
```

The fresh fingerprinted run selected 599 recipes and completed in 652.6 seconds:

| DSL | CAPTURED | PROFILE_SKIPPED | UNSUPPORTED | LOWERING_ERROR | TIMEOUT | IMPORT_ERROR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TileLang | 95 | 51 | 146 | 292 | 1 | 0 |
| Triton | 12 | 0 | 2 | 0 | 0 | 0 |

These counts have distinct meanings:

- 292 TileLang recipes failed in the pinned compiler's native lowering or strict reference/capsule parity; compiler/transform fixtures that are not statically identifiable as expected failures remain visible here rather than being labeled from historical outcomes.
- 51 TileLang recipes were outside the selected parsing profile: 44 required multiple CTAs and seven lowered to no device kernel.
- 146 TileLang raw-IR cases reached capture and failed closed. The largest recognizer classes were unknown nested expression effects (41), unresolved static `Bind` values (24), unresolved participants (22), dynamic control (15), and unsupported barrier/address/attribute forms. Unknown dtype, alias, proxy, region, and TMA evidence also remain explicit failures.
- One TileLang native worker exceeded 300 seconds. No PGIR parser exceeded its isolated 60-second timeout.
- The two Triton failures both used multiple external pointer bases without SHA-bound alias evidence. Scalar single-pointer accesses and statically certified empty protocol graphs are captured.

In total, 302 recipes produced 303 raw device-IR units. The 107 captured recipes emitted 108 strict snapshots and 108 normalized PGIR units because one recipe contains two device kernels. `pipeguard.checker` remained unloaded throughout the runner. Repeating the command validated every fingerprint and artifact hash and resumed all 599 results in 4.75 seconds.

## Checker verification

Every normalized PGIR unit was checked separately with the maximal engine after frontend capture:

| DSL | SAFE | DATA_HAZARD | SYNC_HAZARD | Timeout/error |
| --- | ---: | ---: | ---: | ---: |
| TileLang | 72 | 22 | 2 | 0 |
| Triton | 12 | 0 | 0 | 0 |

All 12 Triton cases also matched the exhaustive engine under `crosscheck`. The largest TileLang case contains 1,048,576 events: loading took 22.28 seconds and maximal checking took 42.58 seconds, producing a deterministic `DATA_HAZARD` witness. Five 262,144-event cases each loaded in about 5.1--5.9 seconds and checked in about 8.0--8.2 seconds. The memory-only fast path records one canonical transition per event and analytically equivalent singleton held-anchor boundaries; protocol-bearing fixtures continue through incremental held-anchor analysis and reduced exhaustive crosschecks.
