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

The initial corpus flow stops after PGIR parsing. It does not invoke `pipeguard.checker` or produce a SAFE/UNSAFE verdict.

Bulk generated TIR/TTGIR, snapshots, normalized PGIR, logs, and progress files live under `generated/` and are intentionally excluded from Git. Source indexes, capsules, manifests, and reviewed recipes are tracked.

## Pinned inventory and recipe coverage

The deterministic source inventory at these locks contains:

| DSL | Source files | Candidates | Lowerable recipes |
| --- | ---: | ---: | ---: |
| TileLang | 643 | 2,054 | 592 |
| Triton | 135 | 802 | 4 |

TileLang has 590 conservative static recipes for dependency-closed, non-CPU/non-AMD PrimFuncs plus two reviewed recipes. Triton has three reviewed recipes and one statically inferred recipe backed by an explicit `ASTSource` payload and literal `GPUTarget("cuda", 90, 32)` compile site. Missing Triton signatures, constexprs, options, or targets are not guessed.

## Frontend-only full run

Run the current concrete recipe set with a 150-second dual-DSL heartbeat:

```bash
/usr/bin/python3 tools/upstream_run.py \
  --dsl all \
  --tilelang-root /mnt/project_dft/xuwen/PipeGuard-asplos/tilelang \
  --triton-root /mnt/project_dft/xuwen/PipeGuard-asplos/triton \
  --output corpus/upstream/generated/parse-runs-final \
  --jobs 6 \
  --progress-interval 150 \
  --timeout 300 \
  --parse-timeout 60
```

The final fingerprinted run selected 596 recipes and completed in 817 seconds:

| DSL | CAPTURED | PROFILE_SKIPPED | UNSUPPORTED | LOWERING_ERROR | TIMEOUT | IMPORT_ERROR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TileLang | 68 | 51 | 155 | 311 | 7 | 0 |
| Triton | 1 | 0 | 3 | 0 | 0 | 0 |

These counts have distinct meanings:

- 311 TileLang recipes failed in the pinned compiler's native lowering or strict reference/capsule parity; this group includes compiler and transform negative tests and is not a frontend parse verdict.
- 51 TileLang recipes were outside the selected parsing profile: 44 required multiple CTAs and seven lowered to no device kernel.
- 155 TileLang raw-IR cases reached capture and failed closed. The largest recognizer classes were nested expression effects (51), unresolved static `Bind` values (24), unresolved participants (22), dynamic control (14), and symbolic regions (10).
- Six TileLang snapshots reached the PGIR adapter but exceeded its isolated 60-second timeout; one additional TileLang worker exceeded the 300-second native-lowering timeout.
- Two Triton cases required explicit external-pointer alias evidence, and one had unresolved ordinary scalar-pointer provenance.

In total, 277 recipes produced 278 raw device-IR units, 75 strict snapshots were emitted, and 69 recipes completed normalized PGIR parsing. `pipeguard.checker` remained unloaded throughout. Repeating the command validated every fingerprint and artifact hash and resumed all 596 results in 6.2 seconds.
