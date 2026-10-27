# Suite 2 Admission Gates

Every cell of the labeled Suite 2 (2 DSL x 7 primitives x 3 result classes x
5 variants = 210 cases, `suite/suite2/manifest.json`) must clear three gates
before its labels are citable. A cell is the 5-variant group keyed by
`(dsl, primitive, result_class)`. The gates run over a full-recapture output
directory (`<run-dir>/cases/<case_id>/` as produced by
`tools/run_suite2_selected210.py`), never over the case sources alone: what is
audited is what was actually captured and compiled.

## Gate 1 — Structural fingerprint uniqueness

Tool: `tools/suite2_structural_audit.py`

Each case is reduced to a structural fingerprint of its captured protocol:

```
material = "ops="            + ",".join(op types of snapshot["operations"])   # order preserved
         + "|actors="        + str(len(snapshot["actors"]))
         + "|resource_kinds="+ ",".join(sorted(kind of snapshot["resources"]))
fingerprint = sha256(material)
```

The op type is the first non-empty of each operation's `op` / `operation` /
`type` / `kind` field. The recipe lives in one documented function
(`fingerprint_material`) and deliberately **excludes shape-level data**:
region offsets and extents, transfer byte counts, per-op instance lists,
barrier arrival counts, and evidence/id fields. Two variants that differ only
in those are the same protocol skeleton and collapse to one fingerprint —
that is the degeneracy this gate exists to catch.

Gate: a cell PASSes when `unique_fingerprints == n` (every variant is a
distinct protocol structure). `--min-unique N` relaxes the bar;
`--min-unique 2` isolates the fully-degenerate cells (all five variants one
skeleton). `--baseline old-report.json` diffs against a previous audit so the
redesign's progress is tracked cell by cell. Nonzero exit on any FAIL.

Status at the `suite2-210-freeze-15835c2b-v1` recapture: the suite FAILS this
gate, as expected. Nine cells are fully degenerate (`unique == 1`):
Memory x all 3 classes x both DSLs (6), Barrier SYNC_HAZARD x both DSLs (2),
and tilelang TMA SYNC_HAZARD (1). Under the strict default, seven further
cells FAIL with one colliding variant pair each (tilelang Barrier
DATA_HAZARD/SAFE, tilelang TMA DATA_HAZARD/SAFE, tilelang mbarrier
DATA_HAZARD, triton Barrier SAFE, triton mbarrier SAFE) — pairs that differ
only in sizes/offsets. The parallel redesign of these cells must bring the
strict audit to 42/42 PASS before re-freeze.

## Gate 2 — Label grounded in lowered CUDA/TTGIR

Tool: `tools/suite2_cuda_grounding.py`

Verifies that the construct named by the label demonstrably **survived
compilation**, by matching per-(primitive, result_class) REQUIRED and
FORBIDDEN regex patterns over the lowered artifact: `kernel.cu` (final CUDA)
for tilelang, `raw.ttgir.mlir` (TTGIR) for triton. The default table is
built in (inspect with `--dump-config`); `--config rules.json` overrides per
family. Matched evidence line numbers are recorded per pattern. Statuses are
PASS / NEEDS_REVIEW / FAIL; a pattern miss or forbidden hit degrades to
NEEDS_REVIEW unless the config marks that pattern `strict`. Nonzero exit on
FAIL (add `--fail-on-review` to treat NEEDS_REVIEW as failing).

What is checkable at text level, per family (and what is not):

| Family | Checkable in lowered artifact | Not checkable at text level |
| --- | --- | --- |
| Barrier (tilelang) | SAFE/SYNC_HAZARD contain `__syncthreads(`; DATA_HAZARD advisory-forbids it | whether a present barrier sits *between the conflicting accesses* (ordering/position), whether a SYNC_HAZARD barrier is divergently guarded |
| Barrier (triton) | SAFE contains a `ttg.barrier` op; SYNC_HAZARD contains the divergent `bar.sync` inline asm; DATA_HAZARD advisory-forbids `ttg.barrier` | same position/divergence caveats |
| Memory (both) | loads/stores survive (`tt.load`/`tt.store`, kernel entry); SYNC_HAZARD's misplaced barrier survives | address overlap/disjointness — the actual hazard — is numeric, not textual |
| mbarrier (tilelang) | `Barrier::init/arrive/wait` + `tl::fence_barrier_init` intrinsics present | arrival-count correctness, phase/parity pairing |
| mbarrier (triton) | `ttng.init_barrier/arrive_barrier/wait_barrier` (DATA_HAZARD family: `barrier_expect` + async TMA copy) | which thread arrives/waits, generation reuse |
| TMA (both) | `tl::tma_load` + `expect_transaction` / `ttng.async_tma_copy_global_to_local` + `ttng.barrier_expect` + init/wait | expect-tx byte count vs transfer size, wait-before-read ordering |
| cp.async (both) | issue/commit/wait intrinsics (`tl::cp_async_*` / `ttg.async_*`) | wait-group depth vs in-flight groups (the actual hazard) |
| WGMMA (both) | `tl::wgmma_*` + `warpgroup_fence_operand` / `ttng.warp_group_dot(_wait)` | wait-group count, operand fence placement relative to accumulator reuse |
| Proxy (tilelang) | SAFE requires `tl::fence_proxy_async`; DATA_HAZARD advisory-forbids it; SYNC_HAZARD lowers via the wgmma pipeline | fence *placement* relative to the generic-proxy write and async-proxy read |
| Proxy (triton) | SAFE requires `ttng.fence_async_shared` + TMA store ops; DATA_HAZARD advisory-forbids the fence; SYNC_HAZARD lowers via wgmma | same placement caveat |

**Honesty statement.** This gate is a text-level survival check and nothing
more. It establishes that the labeled construct was not compiled away,
inlined beyond recognition, or silently replaced; it does **not** establish
that the SAFE/DATA_HAZARD/SYNC_HAZARD semantics hold, because those depend on
positions, address arithmetic, guard predicates, and counts that regex
matching cannot decide. A FORBIDDEN hit (e.g. a `__syncthreads()` appearing
in a DATA_HAZARD kernel) therefore defaults to NEEDS_REVIEW — the barrier may
legitimately sit outside the conflicting pair — and a human (or the checker)
must adjudicate. The **authoritative grounding is the capture snapshot
(`snapshot.json`, which records the operation stream the checker actually
reasons over, with per-op lowering evidence) together with the execution
evidence of Gate 3.** Gate 2 exists to catch the failure mode where source,
snapshot, and label agree but the lowered kernel no longer contains the
phenomenon being labeled.

## Gate 3 — Execution evidence recorded

Every admitted case must carry execution evidence from real hardware: the
NVIDIA Compute Sanitizer sweep ledger produced by
`tools/run_suite2_nvidia_all.py` (`ledger.json`, schema
`pipeguard-suite2-nvidia-all-v1`), which runs every manifest row under both
`racecheck` and `synccheck` and records detection on hazard rows,
false positives on SAFE rows, and timeout behaviour. A case with no sweep
record (or with an unexplained sanitizer/label disagreement that the
adjudication ledger does not resolve) is not admitted.

## Unified rerun pipeline

```
1. Recapture      tools/run_suite2_selected210.py --output <run-dir>
2. Gate 1 (audit) tools/suite2_structural_audit.py --run-dir <run-dir> \
                      --out <run-dir>/structural-audit.json \
                      [--baseline <previous-audit.json>]
                  # must exit 0 (strict: unique == n in all 42 cells)
3. Gate 2 (ground) tools/suite2_cuda_grounding.py --run-dir <run-dir> \
                      --out <run-dir>/grounding-report.json
                  # must exit 0; NEEDS_REVIEW cases require written adjudication
4. Gate 3 (exec)  tools/run_suite2_nvidia_all.py --output <sweep-dir>
                  # ledger.json is the execution-evidence record cited per case
```

The audit runs immediately after recapture (it needs only snapshots + specs);
grounding runs after the audit (same artifacts plus lowered kernels); the
nvidia sweep ledger is the execution-evidence record. Reports are
deterministic sorted-key JSON, suitable for freezing alongside the run.

## Labeling policy

Labels follow the **CUDA memory model**:

* Same-address writes that are unordered are **DATA_HAZARD even when they
  write the same value** — same-value unordered same-address writes are
  DATA_HAZARD. "It happens to produce the right answer" is not SAFE.
* Verdict scope is **per-CTA**. Cross-CTA conflicts are **out of verdict
  scope**: they do not flip a label, and a case whose launch exposes them
  carries a `CROSS_CTA_OUT_OF_SCOPE` warning (emitted by the capture layer,
  see `tools/tilelang_capture.py`) rather than a verdict.
