# Suite 2: native TileLang/Triton frontend pairs

Suite 2 re-implements every labeled semantic case from the direct-PGIR suite in
both DSL frontends. It does **not** construct snapshots or PGIR by changing a
producer field.

For each label, the flow is:

```text
TileLang source factory -> native TileLang lowering -> device TIR
                         -> tilelang_capture -> TileLang snapshot -> PGIR

Triton/Gluon source factory -> native Triton compile-only lowering -> TTGIR
                             -> triton_capture -> Triton snapshot -> PGIR
```

Generated TIR, TTGIR, snapshots, PGIR, cubins, and run outputs are persistent
local artifacts and must not be committed.

## Size

The suite mirrors all 290 direct cases:

```text
290 semantic labels
x 2 native frontend implementations
= 580 compile/capture inputs
```

Complete BUG/CORRECT pairs retain the same `pair_id`, rule, shape, and frontend
parameters. Focused one-sided controls retain their Suite 1 polarity and stage. Shapes are intentionally small:

```text
elements: 1, 2, 4, or 8
pipeline stages: 2, 3, or 4
threads: 32 or 64; WGMMA uses the required 128-thread warpgroup
single CTA only
```

## Required comparison

For every label:

1. both native source implementations are attempted without launching a GPU kernel;
2. source-representable cases must produce evidence-complete snapshots and PGIR;
3. model-stage malformed cases must be rejected with the expected code when the
   DSL can preserve the malformed condition;
4. jointly checked TileLang and Triton cases must agree on verdict;
5. every `CHECKED` or expected `MODEL_REJECTED` result must agree with Suite 1.

`SOURCE_UNREPRESENTABLE`, unsupported native lowering/capture, adapter rejection,
and expected model rejection are reported separately and are never counted as
SAFE. Successfully converted modules are persisted under
`pgir/<dsl>/<label>.json`.

## Persistent local artifacts

Native outputs are written under the repository-local, ignored directory:

```text
.artifacts/frontend-pairs/
```

Run the two persistent compiler processes with:

```text
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
  triton-pg/bin/python tests/suite/frontend_pairs/runners/run_triton_batch.py \
  --output .artifacts/frontend-pairs

tl-pg/bin/python tests/suite/frontend_pairs/runners/run_tilelang_batch.py \
  --output .artifacts/frontend-pairs
```

The runners update results incrementally and accept repeated `--family`,
`--variant`, and `--label` filters. They never launch a GPU kernel. Analyze captured snapshots
with:

```text
/usr/bin/python3 tests/suite/frontend_pairs/runners/analyze_captures.py \
  --artifacts .artifacts/frontend-pairs
```

Suite 2 uses the production `maximal` checker because even small TileLang source
kernels may materialize dozens of lane events. Exhaustive crosscheck remains in
Suite 1, whose direct reference cases are intentionally tiny.

## Latest local result

The persisted 290-label recovery run currently contains:

```text
TileLang: 277 CHECKED, 2 expected MODEL_REJECTED,
          11 SOURCE_UNREPRESENTABLE, 0 UNSUPPORTED, 0 ADAPTER_UNSUPPORTED
          279/279 comparable results match Suite 1
Triton:   275 CHECKED, 1 expected MODEL_REJECTED,
          14 SOURCE_UNREPRESENTABLE, 0 UNSUPPORTED, 0 ADAPTER_UNSUPPORTED
          276/276 comparable results match Suite 1
Paired:   274 jointly checked, 274 matching verdicts
```

`SOURCE_UNREPRESENTABLE` is distinct from extraction failure. The runner still
attempts native lowering and, when possible, persists raw IR, snapshot, PGIR, and
the nearest source-level checker result. The status records why a legal source
program cannot reproduce the direct-PGIR case exactly. For example, TileLang
rejects an out-of-bounds dangling-region approximation, while both native
compilers reject invalid WGMMA shapes and non-shared TMA-store sources. These
cases are not counted as SAFE, checked mismatches, or capture failures.

Other direct-PGIR model tests cannot be reproduced literally by a legal high-level
DSL program. The frontend compiler prevents dangling references, incomplete CTA
participant metadata, invalid cp.async/TMA address spaces, and non-coalesced
polling metadata; it also normalizes barrier invocations and inserts required
WGMMA fences. Their per-DSL limitation is declared in `catalog.py`; the analyzer
never rewrites snapshot metadata to manufacture the direct-PGIR error.

All comparable cases now agree with their Suite 1 expected result. All 280 labels
that are checker-comparable in both DSLs also agree cross-frontend.
