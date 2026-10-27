# Suite "Scalability": checker-core scaling rerun (replaces placeholder figure data)

Rerun of the paper's Suite 4 scalability experiment for the claim:

> "4 DSL/operator series x 5 sizes; PDG construction beyond one million logical
> events; log-log slope ~= 1 (near-linear)."

Measured object: the **checker core only** (`pipeguard.checker.check_module`,
engine `maximal`, pure CPU) — PDG construction plus conflict detection — as a
function of the number of logical events of the input PGIR.

Raw data: [`results-1e640813.json`](results-1e640813.json)
(repo `osdi-rerun` @ `1e640813`). Instances and per-point measurements live in
`instances/` and `measurements/`. Everything was produced by
[`tools/scalability_bench.py`](../../tools/scalability_bench.py).

## 1. Instance construction (and why it is semantically faithful)

The 262 H100 rerun PGIRs (`/mnt/project_dft/xuwen/PipeGuard-osdi-sync/suite4-full-f5d45a2e-{v1,v2}`)
are **not mounted on this machine**, so baselines could not be copied. Instead
every size tier was regenerated locally through the *same frozen suite4
production pipeline*, which is CPU-only end to end:

```
suite/suite4/kernels/<kernel>/sources/original.py        (frozen source capsule)
  -> real compiler lowering, no device codegen, no launch
     TileLang: tilelang.lower(..., enable_device_compile=False)   [tl-pg env]
     Triton:   triton.compile(...).asm["ttgir"]                   [triton-pg env]
  -> frozen frontend capture (tools/tilelang_capture.py / triton_capture.py)
     with the analytically derived TMA-descriptor / binding evidence that
     tools/mutation_suite_worker.py itself uses for suite4
  -> tools/upstream_parse.py -> normalized pgir.json               [checker env]
```

We deliberately did **not** amplify PGIRs by textually replicating events.
Instead, the suite4 workload recipe
(`recipes/<kernel>-workload-original.json`) is re-materialized with a scaled
value of the single binding that controls the number of **pipeline iterations
of the selected CTA** (launch grid and per-iteration structure unchanged).
Each tier is therefore the true PGIR of the real kernel at that workload:
mbarrier generations, wait parities, commit-group sequencing and
TMA/CP_ASYNC rewiring across the added iterations come from the actual
compiler + frontend, not from a replication heuristic.

Fidelity gates, all recorded in `instances/*/*/instance.json`:

1. PGIR schema + semantic validation on load (`pipeguard.load_module`).
2. Every tier of every series must check `SAFE` (all 25/25 do).
3. On the smallest tier of each series, `engine=crosscheck` compares the
   compact parametric representation against the fully expanded reference:
   `SAFE/SAFE` agreement for tilelang-softmax, triton-layer-norm,
   triton-softmax. For the two TMA series (dense-gemm, fused-attention)
   `expand_reference()` fails with a pre-existing checker limitation
   ("explicit engine ordering must reference an ASYNC_ENGINE actor"), so this
   gate is recorded as `UNAVAILABLE` there — reported as-is, not masked.

### Series (4 operator families x 2 DSLs, 5 series, 5 tiers each)

| series | DSL | family | scaled binding (pipeline-iteration dimension) | tiers | events (t0 -> t4) |
|---|---|---|---|---|---|
| tilelang-dense-gemm | TileLang | DENSE_GEMM | K = 256 ... 32768 (iters = K/32, 3-stage pipeline, TMA+MMA) | 5 | 19,622 -> 1,324,166 |
| tilelang-fused-attention | TileLang | FUSED_ATTENTION | seq_kv = 128 ... 2048 (KV-block iters, 2-stage, causal, TMA) | 5 | 60,947 -> 1,608,331 |
| tilelang-softmax | TileLang | SOFTMAX | N = 1024 ... 262144 (online-softmax iters = N/1024) | 5 | 9,600 -> 2,424,960 |
| triton-layer-norm | Triton | LAYER_NORM | N = 1024 ... 262144 (3 masked loops, BLOCK=1024) | 5 | 6,146 -> 1,572,866 |
| triton-softmax | Triton | SOFTMAX | M rows = 1024 ... 262144 (persistent kernel, grid fixed 256) | 5 | 8,192 -> 2,097,152 |

All five series cross 10^6 logical events at the top tier (max 2.42M). Family
substitution note: the suite4 **tilelang**-layer-norm original variant
reproducibly checks `DATA_HAZARD` on this branch (unordered multi-lane writes
to `global:Mean`; same verdict as the local full-corpus rerun), so it cannot
serve as a SAFE baseline; the LAYER_NORM family therefore uses the
**triton**-layer-norm kernel ("choose from the two DSLs" per the suite
design). The grouped-gemm family was not needed to reach 4 families.

## 2. Measurement protocol

- Machine: 128x Intel Xeon Platinum 8362 @ 2.80 GHz, 463 GiB RAM, Linux
  5.4.250 (velinux). **Shared multi-tenant host; no GPU used anywhere.**
- Checker interpreter: `/home/xxw/PipeGuard-sort/envs/checker/bin/python`
  (CPython 3.12.13), `PYTHONHASHSEED=0`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`.
- One worker process per instance, pinned with `taskset -c 104` (a core
  observed idle; verified via `os.sched_getaffinity` in each record).
- Per point: 1 untimed warmup + **5 timed repetitions**, `gc.collect()`
  between runs; the **median** is reported, all raw samples are kept in the
  results JSON.
- Recorded per point: logical events, actors, PDG-construction time
  (`implicit_pdg_build_s` = setup + protocol schedule + boundary
  finalization), conflict-detection time (`race_scan_s`), total check wall
  time, peak RSS (`ru_maxrss` of the worker), machine-load snapshot
  (loadavg before/after, system busy fraction from `/proc/stat` deltas).
- "Events" is the checker's own `statistics.events` (= logical events of the
  lane-parameterized program; identical to `logical_events`).

### Noise

Host is shared with other tenants: 1-minute loadavg ranged 1.08-3.27 (of 128
cores) across the campaign; system-wide busy fraction during each pinned run
was <= 2.9%. Max-min spread of the 5 wall-time samples was <= 0.8% of the
median for all points except triton-softmax t4 (2.9%). Repetitions + median +
core pinning are the noise countermeasures; no point was discarded.

## 3. Results

Log-log OLS fits of median time vs. logical events (5 points per series,
~2 decades of scale; CI95 = two-sided Student-t on the slope, dof=3).

### PDG construction (`implicit_pdg_build_s`)

| series | slope | CI95 | R^2 |
|---|---|---|---|
| tilelang-dense-gemm | **1.109** | [1.077, 1.140] | 0.9998 |
| tilelang-fused-attention | **1.044** | [1.024, 1.065] | 0.9999 |
| tilelang-softmax | **1.065** | [1.048, 1.083] | 0.9999 |
| triton-layer-norm | **1.136** | [1.076, 1.196] | 0.9992 |
| triton-softmax | **1.145** | [1.092, 1.197] | 0.9994 |

### Conflict detection (`race_scan_s`) and total check time

| series | race-scan slope (CI95) | total-check slope (CI95) |
|---|---|---|
| tilelang-dense-gemm | **1.600** [1.302, 1.899] | **1.497** [1.245, 1.749] |
| tilelang-fused-attention | **1.510** [1.334, 1.685] | **1.390** [1.219, 1.560] |
| tilelang-softmax | 0.992 [0.954, 1.031] | 1.058 [1.039, 1.078] |
| triton-layer-norm | 1.051 [1.000, 1.101] | 1.085 [1.041, 1.130] |
| triton-softmax | 1.074 [1.036, 1.111] | 1.087 [1.048, 1.126] |

Per-point medians (seconds; RSS = worker peak):

| series / tier | events | actors | pdg_s | race_s | total_s | RSS MB |
|---|---|---|---|---|---|---|
| dense-gemm k256 | 19,622 | 256 | 0.125 | 0.217 | 0.349 | 48 |
| dense-gemm k1024 | 50,438 | 256 | 0.363 | 0.591 | 0.982 | 86 |
| dense-gemm k4096 | 173,702 | 256 | 1.499 | 3.686 | 5.294 | 258 |
| dense-gemm k16384 | 666,758 | 256 | 6.456 | 42.25 | 49.25 | 1,284 |
| dense-gemm k32768 | 1,324,166 | 256 | 13.20 | 173.3 | 187.8 | 3,484 |
| fused-attention kv128 | 60,947 | 256 | 0.525 | 0.752 | 1.288 | 86 |
| fused-attention kv256 | 163,867 | 256 | 1.433 | 2.815 | 4.287 | 182 |
| fused-attention kv512 | 370,219 | 256 | 3.380 | 8.790 | 12.27 | 375 |
| fused-attention kv1024 | 782,923 | 256 | 7.592 | 30.06 | 37.88 | 769 |
| fused-attention kv2048 | 1,608,331 | 256 | 15.76 | 107.5 | 123.8 | 1,639 |
| tl-softmax n1024 | 9,600 | 128 | 0.056 | 0.007 | 0.064 | 40 |
| tl-softmax n4096 | 38,016 | 128 | 0.230 | 0.025 | 0.261 | 59 |
| tl-softmax n16384 | 151,680 | 128 | 1.017 | 0.098 | 1.141 | 160 |
| tl-softmax n65536 | 606,336 | 128 | 4.579 | 0.410 | 5.128 | 568 |
| tl-softmax n262144 | 2,424,960 | 128 | 19.89 | 1.686 | 22.21 | 2,205 |
| tr-layer-norm n1024 | 6,146 | 128 | 0.014 | 0.024 | 0.038 | 39 |
| tr-layer-norm n4096 | 24,578 | 128 | 0.056 | 0.098 | 0.156 | 50 |
| tr-layer-norm n16384 | 98,306 | 128 | 0.268 | 0.414 | 0.692 | 109 |
| tr-layer-norm n65536 | 393,218 | 128 | 1.458 | 1.680 | 3.206 | 353 |
| tr-layer-norm n262144 | 1,572,866 | 128 | 6.947 | 8.426 | 15.63 | 1,329 |
| tr-softmax m1024 | 8,192 | 128 | 0.018 | 0.094 | 0.113 | 43 |
| tr-softmax m4096 | 32,768 | 128 | 0.077 | 0.385 | 0.465 | 63 |
| tr-softmax m16384 | 131,072 | 128 | 0.390 | 1.665 | 2.073 | 156 |
| tr-softmax m65536 | 524,288 | 128 | 2.095 | 8.053 | 10.23 | 549 |
| tr-softmax m262144 | 2,097,152 | 128 | 9.811 | 35.12 | 45.28 | 2,119 |

## 4. Verdict on the "near-linear" claim

**Supported (with a caveat): PDG construction.** All five series fit with
slope 1.04-1.14 over ~2 decades up to 1.3-2.4M logical events (R^2 >= 0.999).
This is near-linear, matching the checker's own `O(TN)` complexity manifest —
but strictly, slope = 1.0 lies *outside* the 95% CI for four of five series
(only fused-attention's CI, [1.02, 1.06], comes close). Honest reading:
mildly superlinear, consistent with n^(~1.1) or n * polylog(n), not exactly
n^1.

**Not supported for end-to-end check time on the GEMM-style TMA series.**
Conflict detection is the dominant cost at scale for dense-gemm and
fused-attention and is clearly superlinear there: race-scan slopes 1.60 and
1.51, dragging total check time to slopes 1.50 and 1.39. The mechanism is
visible in the data: `race_candidate_pairs` grows ~n^1.8 (dense-gemm:
54,168 pairs at 19.6k events -> 92,224,672 pairs at 1.32M events), because
every added pipeline iteration adds global-memory transfer windows over the
same A/B/C allocations, and candidate pairs over those windows grow close to
quadratically. At 1.3M events the dense-gemm check takes ~188 s, of which
~92% is the race scan. For softmax/layer-norm-style series (zero candidate
pairs) total check time is near-linear (slopes 1.06-1.09).

If the paper figure plots *PDG construction* alone, slope ~= 1 is a fair
summary (1.04-1.14 measured). If it plots *total checker time* for all four
operator families, the claim does not hold as stated: two of four families
are ~n^1.4-1.5 end to end in this rerun.

Peak memory grows near-linearly in events for the softmax/layer-norm series
(~0.9 KB/event) and faster for dense-gemm at the top tier (3.4 GB at 1.32M
events, driven by the materialized candidate-pair structures).

## 5. Reproduction

```bash
REPO=checker/PipeGuard
TL=/home/xxw/PipeGuard-sort/envs/tl-pg/bin/python        # TileLang lowering env
TR=/home/xxw/PipeGuard-sort/envs/triton-pg/bin/python    # Triton lowering env
CK=/home/xxw/PipeGuard-sort/envs/checker/bin/python      # checker env

export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=
cd $REPO
$TL tools/scalability_bench.py generate --series tilelang-dense-gemm \
    --series tilelang-softmax --series tilelang-fused-attention
$TR tools/scalability_bench.py generate --series triton-layer-norm --series triton-softmax
$CK tools/scalability_bench.py parse          # snapshot -> pgir.json (+ t0 crosscheck)
$CK tools/scalability_bench.py measure --cpu <idle-core> --repeats 5
$CK tools/scalability_bench.py report         # -> suite/scalability/results-<shortsha>.json
```

Intermediate snapshots for tiers > t0 are deleted after parsing (sha256 and
byte counts retained in `instance.json`); `pgir.json`, raw lowered IR, and
all evidence sidecars are kept, so `measure`/`report` are re-runnable as-is.
