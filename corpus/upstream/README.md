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
| Triton | 135 | 802 | 313 |

TileLang has 583 static profile-specialization recipes plus two reviewed recipes. Static classification rejects 18 PrimFuncs that explicitly call Blackwell `tcgen05` operations and two PrimFuncs defined inside expected-failure scopes. Source-directory names do not substitute for operation evidence: 13 dependency-closed PrimFuncs under AMD/CPU test directories are attempted under the SM90 profile, producing six additional normalized PGIR programs while unknown effects and control still fail closed.

Triton has 294 reviewed recipes and 19 statically inferred recipes. The reviewed set now grows by reading each real launch/warmup/compile context and recording one concrete recipe per statically proven parameter set; multiple parameter sets for one candidate remain separate recipes over one source capsule. The existing launch extractor remains useful as an index and rejection diagnostic, but it is no longer the primary coverage-expansion mechanism. Ordinary tensor addresses, dynamic pruning/hooks, opaque heuristics, unknown grids, aliases, and unknown compiler options remain fail-closed rather than being filled from observed runtime accidents.

An optional report-only rejection taxonomy keeps extractor limitations separate from immutable corpus evidence. Before applying reviewed overlays, automatic analysis partitions the corpus into 20 candidates with inferred recipes, 84 helpers, 180 unclassified candidates, 30 unresolved autotune roots, 441 ordinary runtime-parameter roots, 16 all-constexpr roots, and 31 zero-argument roots. Reviewed batches 1 through 11 plus the corrected dependency slice raise the tracked source index to 102 candidates with 313 recipes. The slicing correction restores one previously inferred test_tuple.py recipe whose helper parameters had been misclassified as unresolved; it does not infer launch values. At relevant call sites, the largest automatic-rejection classes remain unknown runtime values, unknown grids, non-single-CTA grids, unknown autotune config construction, and dynamic autotune effects. The report lives under ignored `generated/` output and does not alter manifests, recipe IDs, or fingerprints.

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

## AI-reviewed Triton curation batch 3

The third batch expands the static pytest parameter list for `test_dtype_tensor.dtype_tensor_kernel` into 13 independent recipes over one source capsule: `int8`, `int16`, `int32`, `int64`, `int1`, `uint8`, `uint16`, `uint32`, `uint64`, `fp16`, `bf16`, `fp32`, and `fp64`. The list is taken directly from locked Triton's `SINT_TYPES + UINT_TYPES + STANDARD_FP_TYPES`; no runtime dtype is guessed.

The worker reconstructs each compiler constexpr only from an explicit `{"__pipeguard_triton_dtype__": name}` marker. All 13 specializations pass source-derived capsule replay parity, strict capture, PGIR normalization, maximal checking, and exhaustive crosscheck as certified empty protocol graphs with `SAFE` verdicts. A fresh Triton-only run over all 48 current recipes completed in 192.1 seconds with 35 `CAPTURED` and 13 `UNSUPPORTED`; an immediate repeat resumed 48/48 SHA-verified results in 0.262 seconds.

## AI-reviewed Triton curation batch 4

The fourth batch reviews every explicit triton.compile(ASTSource(...)) path in python/test/unit/language/test_compile_errors.py. It adds 15 concrete SM90 specializations: seven direct no-error/warning cases, two finite constexpr-return branches (N=0 and N=1), three SM90-supported FP8 dtypes (fp8e5, fp8e4b15, and fp8e4nv), and three correctly consumed must_use_result cases. All 15 pass source/capsule native TTGIR parity, strict capture, PGIR normalization, maximal checking, and exhaustive crosscheck as certified empty protocol graphs with SAFE verdicts.

The review does not turn expected compiler failures into positive corpus entries. Rejected paths cover undefined names, invalid unary/binary/builtin operations, failed static assertions, failing nested calls, runtime-dependent return shapes, non-power-of-two shapes, forbidden global access, unsupported SM90 FP8 variants, invalid CUDA dot sizes, excessive imprecise accumulation, discarded must_use_result values, constexpr plus do_not_specialize, invalid scaled-dot shapes, nested function definitions, and unsupported histogram input widths. The HIP-only successful minimum-dot branch is outside the selected NVIDIA SM90 target.

This batch also fixes dependency slicing so parameters local to selected helper functions and aggregate-class methods are not recorded as unresolved outer bindings. Besides enabling the three reviewed aggregate/helper cases, deterministic regeneration restores one existing inferred test_tuple.py recipe. A fresh Triton-only run over all 64 recipes completed in 107.1 seconds with 51 CAPTURED and 13 UNSUPPORTED; all 51 captured modules matched maximal and exhaustive checking with SAFE, and an immediate SHA-verified repeat resumed 64/64 results in 0.352 seconds.

## AI-reviewed Triton curation batch 5

The fifth batch reviews explicit compiler paths in test_compile_only.py, test_perf_warning.py, runtime/test_subproc.py, and test_warp_specialization.py. Six SM90 maxnreg specializations enumerate the complete static instrumentation_mode list; all six pass native parity and strict capture, contain one event each, and match maximal/exhaustive checking with SAFE verdicts. Two CUDA performance-diagnostic specializations pass native parity but fail closed during capture: one at an unsupported ttg.global_scratch_alloc operation and one because three external pointer bases lack alias evidence.

SM100 and SM80 calls, HIP calls, raw IR file inputs, and subprocess calls whose active target is not statically constrained to SM90 do not receive SM90 JIT recipes. Existing top-k and SM90 FP8 compile-only recipes are retained rather than duplicated. The tracked Triton inventory is now 72 concrete recipes over 52 candidates; focused run artifacts are stored in resumable ignored output directories and are not committed.

## AI-reviewed Triton curation batch 6

The sixth batch closes the remaining zero-argument and all-constexpr roots that have direct successful launches. It adds three gpuhello BLOCK_SIZE=1024 specializations with a statically evaluated 97-CTA grid, four print-helper parameter sets, and the successful static_assert COND=True specialization. Native replay reports one CAPTURED and SAFE crosscheck, four PROFILE_SKIPPED finite multi-CTA launches (three grid=97 and one grid=2), and three explicit tt.print unsupported operations.

Capsule extraction now strips only syntactically explicit pytest metadata decorators while preserving Triton compilation decorators and original candidate identity. This removes pytest as a false runtime dependency without installing it in the Triton environment. Plugin-hook-dependent kernels, expected compiler failures, and constexpr-assignment cases requiring patch_kernel source rewriting plus a callable constexpr_type binding remain explicit reviewed gaps rather than guessed recipes.

## AI-reviewed Triton curation batch 7

The seventh batch records 17 real tutorial parameter sets from low-memory dropout, layer normalization, libdevice asin, and programmatic dependent launch. Static wrapper inputs recover exact fp32/fp16/int32 signatures, constexpr blocks, layernorm heuristics, PDL modes, and grids, including five benchmark sizes for each PDL mode. The customized machine-relative libdevice path is not persisted; only the default-libdevice specialization is recorded.

All 17 pass native source/capsule replay parity. Fourteen finite multi-CTA launches are PROFILE_SKIPPED with exact grids (64, 97, 1151, or benchmark-dependent 8192 through 131072). The two single-CTA dropout kernels and the size-1024 PDL kernel fail closed because their multiple external pointer bases lack explicit alias evidence. No alias relation is inferred from allocation code or observed addresses. Focused artifacts for all 17 corrected recipe IDs are retained in resumable ignored output.

## AI-reviewed Triton curation batch 8

The eighth batch reviews runtime/test_launch.py. It adds six concrete recipes: the memory-leak launch, the kernel-load-hook launch, three compiler-option specializations (base, num_warps=1, and enable_fp_fusion=false), and the pre-run-hook kernel with n_elements=4. Machine-relative extern_libs and a callable launch_metadata hook are not serialized, and the existing multiple-hooks recipe is not duplicated.

Four single-CTA recipes are CAPTURED and match maximal/exhaustive checking with SAFE verdicts; the pre-run-hook case contains two memory events. The memory-leak and kernel-load-hook cases pass native parity and are PROFILE_SKIPPED with exact finite grids 10 and 6. Focused artifacts are persisted under corpus/upstream/generated/triton-runtime-launch-b8.

## AI-reviewed Triton curation batch 9

The ninth batch begins the high-density review of `python/test/unit/language/test_core.py`. It adds 47 concrete launch parameter sets over 15 source candidates: all 20 `test_store_constant` dtype/constant-field combinations, four `test_arange` starts, seven static `test_if` modes, four early-return specializations, and direct single-CTA launches for bfloat16 rounding, expand/squeeze, defaults, keyword-only defaults and overrides, constexpr shapes, static ranges, loop temporaries, dtype introspection, and poison return. Source parameters come from the locked pytest tables, Python defaults, and tensor construction sites; no patched-expression kernel is reconstructed.

All 47 pass source/capsule native TTGIR parity. Twenty-six produce PGIR and return `SAFE` under both maximal and exhaustive crosscheck. The remaining 21 fail closed at strict capture: nine unresolved runtime store masks, eight two-base alias obligations, one three-base alias obligation, one `ub.poison`, one unresolved dynamic yield, and one pointer `tt.bitcast`. The persisted fresh run is `corpus/upstream/generated/triton-test-core-b9/run/709a1f22e636f7ee`.

## AI-reviewed Triton curation batch 10

The tenth batch adds another 47 launch parameter sets over 14 `test_core.py` candidates: twelve transpose dtypes, four `umulhi` dtypes, eight permute shape/stride layouts, eight 2-D transpose combinations, six sum dtype/default cases, and direct pointer, interleave, split, cumsum, and non-overflowing boundary-value launches. The interleave decorator closure is explicitly materialized for both `debug` values. The custom callable `repr` used by `test_value_specialization` remains a reviewed gap rather than being replaced by Python's builtin `repr`.

Triton capture now classifies `tt.join`, `tt.split`, and `tt.unsplat` as value-only pure operations only when none of their operands or results carries pointer, memdesc, or tensor-descriptor provenance. Dedicated tests verify a pure join, a split/unsplat chain, and fail-closed pointer join. This restores three interleave recipes without weakening alias analysis. After the rule, 13 batch recipes produce PGIR and crosscheck `SAFE`; the other 34 terminate at explicit two- or three-base alias obligations. Native parity succeeds for all 47. The persisted batch run is under `corpus/upstream/generated/triton-test-core-b10/` and the focused operation-rule run is `corpus/upstream/generated/triton-value-pure-rule/f2d9348c4a318e6e`.

## AI-reviewed Triton curation batch 11

The eleventh batch expands two complete, finite pytest matrices from `test_core.py` into 116 reviewed recipes: all 96 combinations of two dtypes, two shapes, and 24 permutations for `test_trans_4d`, plus all 20 rank/transpose combinations for `test_dot_multidim`. Every recipe uses the source launch's single-CTA grid; the 4-D transpose recipes also retain `num_warps=8`. No autotune winner or runtime value is inferred.

All 116 recipes pass native source/capsule TTGIR parity in the persisted run `corpus/upstream/generated/triton-test-core-b11/run/9da54157795da9ca`. Strict capture rejects all of them explicitly: 96 descriptor-based transpose variants contain unsupported `ttg.global_scratch_alloc`, while the 20 multidimensional dot variants require unresolved alias relations among three external pointer bases. These are frontend/profile gaps after successful native lowering, not missing specialization parameters or parser crashes.

## Triton selected-CTA capture and fresh full run

Triton capture now keeps the source example's complete launch grid while selecting one CTA for analysis. The default coordinate is `(0, 0, 0)`: `tt.get_program_id(axis)` evaluates to that coordinate and `tt.get_num_programs(axis)` evaluates to the corresponding real grid extent. The snapshot and PGIR record both `launch_grid` and `selected_cta`, and the checker manifest scopes `SAFE` to that selected CTA rather than the whole launch. This does not modify runtime signatures, constexprs, compiler options, or launch sizes. Triton CTA clusters with `ttg.num-ctas > 1` remain fail closed.

A fresh six-worker run over all 313 concrete Triton recipes completed in 1000.2 seconds with 105 `CAPTURED` and 208 `UNSUPPORTED`; there are no `PROFILE_SKIPPED`, lowering errors, import errors, or timeouts. All 313 establish native source/capsule TTGIR parity. The 105 captured specializations comprise 101 unit-grid launches, three real 97-CTA launches, and one real `(1, 3, 2)` launch. The 208 capture rejections are explicit: 105 unsupported operations, 88 unresolved alias obligations, nine unresolved predicates, five unresolved SSA values, and one multi-function module. After adding exact dimensions for the two existing multidimensional reviewed grids `(1, 2)` and `(1, 3, 2)`, an immediate repeat SHA-resumed 311/313 artifacts and recomputed only those two recipes in 4.3 seconds with unchanged totals. Generated artifacts remain ignored under `corpus/upstream/generated/triton-selected-cta-full/`.

Five nonempty captured kernels were checked with the maximal engine for 500 repetitions each. All return `SAFE`; all satisfy `scheduler_work_units <= 12 * n_ext * t_eff`, `pdg_edges <= n_ext * (t_eff + 4)`, and one insertion per unique PDG edge.

| Source kernel | Events | PDG build median | Race scan median | Maximal total median |
| --- | ---: | ---: | ---: | ---: |
| `test_keyword_only_arguments.kernel` | 3 | 10.233 us | 20.007 us | 244.625 us |
| `test_compilation_listener.cumsum_kernel` | 2 | 9.140 us | 17.177 us | 217.758 us |
| `test_pre_run_hooks.add_kernel` | 2 | 9.345 us | 17.262 us | 220.032 us |
| `test_static_range.loop_kernel` | 1 | 7.109 us | 12.089 us | 186.185 us |
| `test_cumsum_dtype.kernel` | 1 | 7.243 us | 12.133 us | 193.198 us |

The maximal checker was then run over all 105 captured specializations, with one untimed warmup and 200 measured repetitions per specialization. All 105 return `SAFE` and all 105 satisfy both scheduler and PDG-edge certificate bounds. Together they contain 38 events: 73 specializations have zero events, 28 have one, two have two, and two have three. Across the 105 per-specialization medians, complete maximal checking is 135.850 us median, 188.172 us p95, and 245.030 us maximum; PDG construction is 5.847 us median and 10.316 us maximum; race scanning is 4.719 us median and 20.346 us maximum. The sum of all 105 checker medians is 15.984 ms. The ignored detailed reports are `corpus/upstream/generated/triton-selected-cta-full/checker-all-maximal-v1.json` (SHA-256 `0a7a04d63ac6fe30cfadfe3184129f87ed923655a179388aa1ab9dac483b6b62`) and the equivalent flat CSV `checker-all-maximal-v1.csv` (SHA-256 `7059cfd59905d161896a401be2d7b4ea9f288fa86a4dc3f9247881cd76b3cb15`); each has exactly 105 rows/records with source, recipe, grid, selected CTA, event count/types, verdict, checker statistics, and median/p95 phase timings.

A controlled scale-up of the real `cumsum_kernel` read/write event pattern confirms linear PDG construction and scheduler counters: from 64 to 256 events, PDG edges grow from 63 to 255, scheduler work from 256 to 1024, and median PDG construction from 73.0 to 261.3 us. Race conflict enumeration is a separate output-sensitive phase: when every effect shares one overlapping region, there are inherently `E * (E - 1) / 2` candidate pairs. The same 64-to-256 event scale therefore increases race scanning from 0.529 to 5.899 ms. The `O(TN)` claim applies to the maximal scheduler and direct PDG construction; race scanning is reported separately with its unavoidable worst-case pair count rather than being folded into that bound.

## Triton synchronization-risk and GSan calibration batch

Four source-proven Hopper TMA specializations now complete recipe, native source/capsule TTGIR parity, strict snapshot v2, PGIR, and maximal checking. Runtime scalar evidence is SHA-bound to the unmodified native TTGIR rather than rewriting runtime parameters as constexpr. The capture rules added for these cases are generic: indexed barrier arrays, loop-carried pointer/register provenance, fixed-width integer evaluation, compiler-created tensormaps, exact strided tensor-map footprints, partial-CTA collective barriers, per-owner WGMMA/TMA-store streams, per-observer mbarrier waits, and final-PTX WGMMA waits accepted only with SHA-bound evidence. The default mbarrier invalidation rule follows NVIDIA PTX: a valid but unused initialized barrier may be invalidated; started incomplete phases, in-flight asynchronous work, and outstanding wait users remain errors.

| Official kernel and selected specialization | Events | Race pairs | PDG build | Race scan | Maximal total | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `small_tma_load_kernel`, `N=12`, grid 1 | 21 | 15 | 54.676 us median (50 runs) | 225.191 us median | 4.076 ms median | `SAFE` |
| `matmul_kernel_tma`, `M=N=512,K=128`, grid `(64,1)`, CTA 0 | 493 | 454 | 2.830 ms | 26.302 ms | 7.532 s | `SAFE` |
| `matmul_tma_ws_kernel`, `M=N=2048,K=512`, grid 256, CTA 0 | 2,428 | 15,624 | 20.983 ms | 1.591 s | 224.717 s | `SAFE` |
| `matmul_kernel_descriptor_persistent`, `M=N=8192,K=128`, grid 132, CTA 0 | 2,668 | 49,640 | 8.452 ms | 1.880 s | 7.365 s | `SAFE` |

All four satisfy `scheduler_work_units <= 12 * n_ext * t_eff` and the direct-PDG edge bound. The warp-specialized matmul has `n_ext=4576`, `t_eff=2074`, 79,698,597 scheduler work units, and 17,746 PDG edges. The persistent specialization has `n_ext=2964`, `t_eff=74`, 1,511,621 scheduler work units, and 11,879 PDG edges. Exact tensor-map strides increase event counts by splitting noncontiguous global tiles into contiguous runs; this removes the prior unsound zero-coordinate shortcut. Total time remains dominated by incremental held-anchor replay and output-sensitive race enumeration, not PDG construction. The source-proven 32-cubed specialization remains fail closed because its TMA loads rely on out-of-bounds tensor-map zero fill, whose destination-only zero-fill effect is not representable by the current equal-byte source/destination PGIR event.

A reduced DSL-neutral calibration derived from Triton's official GSan contracts also passes 14/14 checks: the release/acquire GPU-scope handoff is `SAFE` under maximal, exhaustive, and crosscheck; four semantic mismatches, four CTA-scope cross-SM mismatches, and the RAW/WAR/WAW cases return `DATA_HAZARD/UNORDERED_CONFLICT`. These bounded fixtures calibrate the checker semantics but are not reported as full frontend support for the original runtime spin loops. The complete official `_cross_sm_atomic_sync_kernel`, `serialized_add`, and LayerNorm backward lock kernels all establish native source/capsule parity with exact launch, scalar, dtype, and allocation evidence, then fail closed at `tt.atomic_rmw` or `tt.atomic_cas`; their polling iteration counts depend on another CTA and cannot yet be finitely expanded. PDL multi-kernel tests, Gluon cluster tests, and Blackwell-only tests remain outside this batch.

### Current-revision Triton production-pipeline sweep

A follow-up sweep targets locked Triton revision `61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f` itself. Native specialization again uses compile-warmup/FakeTensor and never launches a GPU kernel. The reviewed source scope includes the production `triton_kernels` regular and persistent matmul implementations plus the official fused-attention tutorial's forward and backward pipelines. Concrete cases cover BF16, FP16, MXFP4, and MXFP8 inputs; 4- and 8-warp configurations; `block_m` 16 and 128; plain and batched launches; causal and noncausal attention; WGMMA, cp.async, TMA load/store, mbarrier, proxy fences, and persistent-loop reuse. Real launch grids are preserved and CTA `(0,0,0)` is selected.

All completed maximal checks in this sweep are `SAFE`; no unpatched concurrency bug was found in these current-revision production pipeline paths. The result is deliberately bounded rather than a claim that all 802 function candidates or all Triton kernels are bug-free. Ragged matmul loop bounds depend on runtime slice metadata, TopK sparse write addresses depend on loaded indices, and FP8 scale/bitmatrix paths use atomic RMW operations, so those cases are outside the static/no-atomic profile. TMA boundary zero-fill loads and out-of-bounds-discard stores are represented with isolated synthetic source/sink regions while retaining the complete shared-memory completion footprint; the corresponding small persistent specializations also check `SAFE`.

The sweep also corrected several frontend gaps exposed only by production TTGIR: exact pointer broadcasting, selected `scf.if` and finite `cf.cond_br` provenance, loop-carried async tokens, pure `tt.bitcast`/`tt.fp_to_fp`/`tt.clampf`/`ttg.fp4_to_fp`/`tt.extern_elementwise`/`tt.mulhiui`/`ub.poison`, finite zero-result private `tt.call` expansion, empty WGMMA waits after zero-iteration or fully drained streams, and contiguous cp.async footprint coalescing. SM90 cp.async capacity is 16 with wait thresholds 0--15, matching emitted H100 SASS such as `DEPBAR.LE SB0, 0xc`; the previous capacity-8 assumption produced a false positive on causal attention backward and is not reported as a Triton bug. Generated TTGIR, snapshots, PGIR, and checker reports remain untracked under `/tmp/pipeguard-search/`.

A subsequent launch-root audit defines the useful-kernel denominator explicitly rather than using the 802-function syntactic index. It found 51 direct roots: 28 product roots in `python/triton_kernels/triton_kernels` after excluding the testing-only random-sign utility, plus 23 standard Triton tutorial roots. Twenty-eight roots have static NVIDIA SM90 specializations inside the PipeGuard profile; all 28 are covered by 95 completed maximal checks (84 unique TTGIR modules), and all 95 are `SAFE`. The remaining 23 roots are individually excluded for runtime-loaded address/control dependence, atomic RMW, multi-GPU remote memory, Gluon, AMD, Blackwell/CLC, inter-kernel PDL, or simple profiling-helper status. `triton-official-kernel-coverage.md` records the complete root list, checked specialization classes, and per-root exclusion evidence. Generated aggregate evidence remains at `/tmp/pipeguard-search/official-useful-coverage-records.json` (SHA-256 `a4db09aa8a7f7a92dd575382a9220a63ee5db2184e0af2ed83645cae5928dde7`).

### Checker-detected historical Triton production bug

PipeGuard reproduces the missing-WGMMA-wait bug fixed by Triton PR [#8964](https://github.com/triton-lang/triton/pull/8964) and reapplied by [#8997](https://github.com/triton-lang/triton/pull/8997). This is an ordinary `triton_kernels` production path, not a GSan/ConSan negative fixture: `python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py` at source commit `5784490c92dbf1c82dfe163ac0cdbc6aedab8873`, using the repository's real `test_matmul.py` specialization `M=768,N=512,K=1024`, plain BF16-by-MXFP4, `block_m=16`, persistent mode, Hopper-swizzled weights, `num_warps=8`, `num_stages=3`, grid 96, selected CTA 0. Native specialization used Triton's compile-warmup/FakeTensor path and did not launch a GPU kernel.

The buggy official pass emits two register-source WGMMA operations followed by `wait_group 1`, then consumes the still-live accumulator with `tt.trans` in the persistent epilogue. The complete strict snapshot contains 900 events. Maximal checking reports `DATA_HAZARD/UNORDERED_CONFLICT` between `triton:cta:e00200:op00197` (the second `WgmmaIssue`, whose accumulator write remains live until `RESULT_READY`) and `triton:cta:e00204:op00199` (the epilogue register read). Its certificate has `n_ext=1662`, `t_eff=716`, 1,679 PDG nodes, 3,513 PDG edges, and 8,647,949 scheduler work units; checking took 27.528 seconds. The buggy TTGIR SHA-256 is `71248f0521a0a93964e6386675491b2d6ee821ad9e5f9273e5fa655d968426be`, and the deterministic snapshot SHA-256 is `140d41e2203b174864ba560c5feb8f0a96f4e6a8012e10955bf9febf163b6c9b`.

As a differential check, the same historical source and specialization lowered with the corrected pass contains the missing warp-group-local `wait_group 0` immediately before the epilogue `tt.trans`. Its complete 895-event snapshot checks `SAFE` in 29.477 seconds, with `n_ext=1657`, `t_eff=716`, 1,674 PDG nodes, 3,536 PDG edges, and 8,633,604 scheduler work units. The fixed TTGIR SHA-256 is `60284dcedce4e157b6acf32ec148188b7fa27d8a1f21340e3644692d785a5e34`, and the deterministic snapshot SHA-256 is `995d3869aeaa02bf033ef19cd5c4a8d44654e4dfd9573420b94a24c7c5ba97f0`. The one semantic difference relevant to the verdict is the result-ready wait; generated TTGIR, snapshots, PGIR, and checker reports remain untracked under `/tmp/pipeguard-search/wgmma-official-bug/` (report SHA-256 `fbca5fb1d3eea20d9b64e0459a2d370036888bd3cfa43cf033ad3ca18037f71a`).

This finding is historical: locked Triton revision `61c02bcd` contains the corrected wait placement and is not reported as vulnerable. It nevertheless supplies a source-proven official GPU-kernel bug in PipeGuard's WGMMA/result-read scope, together with a checker witness and a fixed-version `SAFE` control.

The source audit also confirmed a host-runtime concurrency bug outside PipeGuard's GPU-kernel PGIR scope. At locked revision `61c02bcd`, both `triton.testing.do_bench_cudagraph` and `do_bench_cudagraph_proton` enter a newly created CUDA stream and invoke the benchmark function before waiting for the stream that produced its inputs (`python/triton/testing.py:159` and `:240`). The first warmup or graph capture may therefore consume incompletely initialized buffers. Upstream issue [#10312](https://github.com/triton-lang/triton/issues/10312) independently reports this race for `do_bench_cudagraph`; its proposed synchronization commits are absent from the locked revision. The Proton variant has the same source pattern. This is recorded as a confirmed official-library finding, but not as a checker-detected kernel bug.

The ignored machine-readable report is `corpus/upstream/generated/triton-recommended-detection/report.json` (SHA-256 `fbf0f270b6830b39c3d1e510e7642f3cab8a5a6b23b77608ab137a2699e57fec`). It records event histograms, timings, checker statistics, verdicts, exact fail-closed diagnostics, and source-audit findings without committing generated TTGIR, snapshots, or PGIR.

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

The last full dual-DSL baseline predates reviewed batch 1: it selected 606 recipes and completed in 646.7 seconds. The current inventory contains 898 recipes; its next fresh dual-DSL run follows the focused `test_core.py` checkpoints.

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

All 15 baseline Triton cases matched the exhaustive engine under crosscheck; reviewed batches 1 through 10 and the slicing-restored recipe add 86 independently crosschecked SAFE cases, for 101 current Triton PGIR specializations. The largest TileLang case contains 1,048,576 events: loading took 22.28 seconds and maximal checking took 42.58 seconds, producing a deterministic DATA_HAZARD witness. Five 262,144-event cases each loaded in about 5.1--5.9 seconds and checked in about 8.0--8.2 seconds. The memory-only fast path records one canonical transition per event and analytically equivalent singleton held-anchor boundaries; protocol-bearing fixtures continue through incremental held-anchor analysis and reduced exhaustive crosschecks.
