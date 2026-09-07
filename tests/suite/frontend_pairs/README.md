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

Generated TIR, TTGIR, snapshots, cubins, and run outputs are temporary artifacts
and must not be committed.

## Size

The suite mirrors the 240 direct cases:

```text
240 semantic labels
x 2 native frontend implementations
= 480 compile/capture inputs
```

Each BUG case has one CORRECT counterpart with the same `pair_id`, rule, shape,
and frontend parameters. Shapes are intentionally small:

```text
elements: 1, 2, 4, or 8
pipeline stages: 2, 3, or 4
threads: 32 or 64; WGMMA uses the required 128-thread warpgroup
single CTA only
```

## Required comparison

For every label:

1. both native source implementations must lower without launching a GPU kernel;
2. both capture tools must produce evidence-complete snapshots;
3. both adapters must produce PGIR;
4. TileLang and Triton must agree on verdict and rule-level witness;
5. each result must agree with the direct-PGIR reference case.

Unsupported native lowering or capture is reported separately and is never
counted as SAFE or as a semantic match.
