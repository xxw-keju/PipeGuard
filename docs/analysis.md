# Analysis implementation and validation

## Production paths

`check_module` accepts only maximal. Ordinary PGIR runs through `Executor` and
`IncrementalEngine`. Compact PGIR runs through `DirectParametricChecker` without
expanding the serialized input into per-thread JSON. The CLI's `check`, `explain`,
and frontend `compare` commands all use maximal. `compare` compares frontend
representations, not execution engines.

The expanded scheduler maintains candidate readiness using static ranks and
transition-delta notifications. It performs a canonical run and then held-anchor
passes, reusing state along each anchor stream. Copy completions and deferred
cp.async/mbarrier arrivals are independent synthetic anchors. Their readiness is
updated by issue/completion notifications; no global scan or state-space search
is used as a fallback for those events.

Some protocol-state helpers, such as deterministic stepping and fingerprints,
remain for focused unit tests. There is no runtime all-schedule enumeration or
engine-comparison mode. The integration subprocess provides a shared capture/check
deadline; timeout yields INCONCLUSIVE.

## Validation retained

- Protocol regressions check fixed expected SAFE/hazard outcomes and witnesses.
- Compact and expanded tests exercise each representation through maximal.
- An independent declarative bounded test covers transaction-free mbarrier
  programs and compares expected verdicts and event order with the checker.
- Bounded endpoint/visibility formulas check cp.async, WGMMA, TMA-store, and
  ordinary-memory scenarios without calling scheduler readiness to derive labels.
- Frontend fixtures check exact evidence, normalization, supported operations,
  and explicit unsupported outcomes.
- Integration tests cover wheel installation, evidence binding, timeout cleanup,
  selected-CTA scope, and optional compiler availability.

The declarative bounded test is test code, not an installed execution engine.
Passing finite regressions does not establish general soundness/completeness.
The transaction-free held-anchor argument has narrower premises than the full
SM90 feature set. Known gaps in [semantics](semantics.md) remain and must not be
hidden by treating tests or agreement with labels as a general proof.

## Performance

Compact capture avoids materializing expanded event objects during checking, but
can still carry large per-instance fields. Capture, serialization, graph building,
and conflict enumeration have different costs. The expanded scheduler's work
certificate measures scheduling and boundary construction; it does not bound
address recovery, graph reachability, JSON I/O, or all conflict pairs. Large-input
budgets are analysis limits, not protocol errors.

## Run tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

Compiler-dependent fixtures skip when the corresponding pinned runtime is absent.
Set `TILELANG_PYTHON` and `TRITON_PYTHON` to suitable interpreter paths to enable
those tests. Benchmark datasets and campaign tools are not shipped in this branch.
