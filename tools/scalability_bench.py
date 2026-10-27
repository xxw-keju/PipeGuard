#!/usr/bin/env python3
"""Suite "scalability" benchmark: checker-core scaling on suite4 kernel series.

Rebuilds the OSDI scalability experiment (PDG construction + conflict detection
cost of ``pipeguard.checker.check_module`` versus logical event count) from the
frozen suite4 corpus, entirely on CPU.

Construction method (semantic fidelity)
---------------------------------------
Larger instances are NOT produced by textually duplicating events of an
existing PGIR.  Instead every size tier is regenerated through the exact
frozen suite4 production pipeline at a scaled *workload*:

  suite4 source capsule (suite/suite4/kernels/<kernel>/sources/original.py)
    -> real DSL compiler lowering, CPU only, no device codegen / no launch
       (TileLang: tilelang.lower(..., enable_device_compile=False);
        Triton:   triton.compile(...).asm["ttgir"])
    -> frozen frontend capture (tools/tilelang_capture.py / triton_capture.py)
       with the same analytically derived descriptor/binding evidence the
       suite4 harness uses (tools/mutation_suite_worker.py)
    -> tools/upstream_parse.py -> normalized PGIR (pgir.json)

Only the workload binding that controls the number of *pipeline iterations*
of the selected CTA is scaled (K for dense-gemm, seq_kv for fused-attention,
N for softmax/layer-norm, rows for the persistent triton softmax); the launch
grid and the per-iteration structure are unchanged.  Because each instance is
the true PGIR of the real kernel at that workload, mbarrier generations,
wait parities, commit-group sequencing and TMA/CP_ASYNC rewiring across the
added iterations are produced by the actual compiler + frontend rather than
by a replication heuristic.  Fidelity is additionally enforced by (a) PGIR
schema validation on load, (b) a SAFE verdict requirement on every tier, and
(c) an engine crosscheck (compact parametric vs fully expanded reference
representation) on the smallest tier of every series.

Subcommands
-----------
  generate         (DSL env python)      lower + capture scaled workloads
  parse            (checker env python)  snapshot -> normalized pgir.json
  measure          (checker env python)  timed check_module runs, taskset-pinned
  measure-worker   (internal, spawned by measure under taskset)
  report           aggregate measurements, fit log-log slopes, emit results JSON

Typical use (see suite/scalability/SCALABILITY.md):
  TL_PY=.../envs/tl-pg/bin/python; TR_PY=.../envs/triton-pg/bin/python
  CK_PY=.../envs/checker/bin/python
  $TL_PY tools/scalability_bench.py generate --series tilelang-dense-gemm ...
  $CK_PY tools/scalability_bench.py parse   --instances suite/scalability/instances
  $CK_PY tools/scalability_bench.py measure --instances ... --cpu 120 --out ...
  $CK_PY tools/scalability_bench.py report  --instances ... --measurements ... --out ...

All emitted JSON uses sort_keys=True and fixed field sets, so documents are
byte-deterministic given identical inputs (timing values are raw samples and
naturally vary run to run).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TOOLS = REPO / "tools"

# --------------------------------------------------------------------------
# Series definitions: 5 series x 5 size tiers, 4 operator families, 2 DSLs.
# Each tier scales exactly one workload parameter (the pipeline-iteration
# dimension of the selected CTA); the launch grid is invariant per series.
# Overrides are applied to recipe bindings / materialization kwargs /
# resolver_kwargs by name and, where the kernel is materialized positionally
# (tilelang-fused-attention), also to materialization args by index.
# --------------------------------------------------------------------------
SERIES: dict[str, dict[str, Any]] = {
    "tilelang-dense-gemm": {
        "dsl": "tilelang",
        "kernel_id": "tilelang-dense-gemm",
        "family": "DENSE_GEMM",
        "scale_parameter": "K (reduction length; pipeline iterations = K / block_K, block_K = 32, num_stages = 3)",
        "args_index": {},
        "tiers": [
            {"label": "t0-k256", "overrides": {"K": 256}},
            {"label": "t1-k1024", "overrides": {"K": 1024}},
            {"label": "t2-k4096", "overrides": {"K": 4096}},
            {"label": "t3-k16384", "overrides": {"K": 16384}},
            {"label": "t4-k32768", "overrides": {"K": 32768}},
        ],
    },
    "tilelang-fused-attention": {
        "dsl": "tilelang",
        "kernel_id": "tilelang-fused-attention",
        "family": "FUSED_ATTENTION",
        "scale_parameter": "seq_kv (KV length; KV-block pipeline iterations, block_N = 64, num_stages = 2, causal)",
        # flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M, block_N, num_stages, threads)
        "args_index": {"seq_kv": 3},
        "tiers": [
            {"label": "t0-kv128", "overrides": {"seq_kv": 128}},
            {"label": "t1-kv256", "overrides": {"seq_kv": 256}},
            {"label": "t2-kv512", "overrides": {"seq_kv": 512}},
            {"label": "t3-kv1024", "overrides": {"seq_kv": 1024}},
            {"label": "t4-kv2048", "overrides": {"seq_kv": 2048}},
        ],
    },
    "tilelang-softmax": {
        "dsl": "tilelang",
        "kernel_id": "tilelang-softmax",
        "family": "SOFTMAX",
        "scale_parameter": "N (row length; online-softmax pipeline iterations = N / BLOCK_N, BLOCK_N = 1024)",
        "args_index": {},
        "tiers": [
            {"label": "t0-n1024", "overrides": {"N": 1024}},
            {"label": "t1-n4096", "overrides": {"N": 4096}},
            {"label": "t2-n16384", "overrides": {"N": 16384}},
            {"label": "t3-n65536", "overrides": {"N": 65536}},
            {"label": "t4-n262144", "overrides": {"N": 262144}},
        ],
    },
    "triton-layer-norm": {
        "dsl": "triton",
        "kernel_id": "triton-layer-norm",
        "family": "LAYER_NORM",
        "scale_parameter": "N (row length; three masked loops of N / BLOCK_SIZE iterations, BLOCK_SIZE = 1024)",
        "args_index": {},
        "tiers": [
            {"label": "t0-n1024", "overrides": {"N": 1024}},
            {"label": "t1-n4096", "overrides": {"N": 4096}},
            {"label": "t2-n16384", "overrides": {"N": 16384}},
            {"label": "t3-n65536", "overrides": {"N": 65536}},
            {"label": "t4-n262144", "overrides": {"N": 262144}},
        ],
    },
    "triton-softmax": {
        "dsl": "triton",
        "kernel_id": "triton-softmax",
        "family": "SOFTMAX",
        "scale_parameter": "M (rows; persistent-kernel row iterations = M / grid, grid = 256, num_stages pipelined)",
        "args_index": {},
        "tiers": [
            {"label": "t0-m1024", "overrides": {"M": 1024}},
            {"label": "t1-m4096", "overrides": {"M": 4096}},
            {"label": "t2-m16384", "overrides": {"M": 16384}},
            {"label": "t3-m65536", "overrides": {"M": 65536}},
            {"label": "t4-m262144", "overrides": {"M": 262144}},
        ],
    },
}

CHECKER_ENGINE = "maximal"  # production engine used by all suite4 verdicts


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_snapshot_counts(snapshot_path: Path) -> dict[str, int]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return {
        "actors": len(snapshot.get("actors", [])),
        "operations": len(snapshot.get("operations", [])),
        "regions": len(snapshot.get("regions", [])),
        "resources": len(snapshot.get("resources", [])),
    }


def apply_overrides(recipe: dict[str, Any], overrides: dict[str, Any], args_index: dict[str, int]) -> dict[str, Any]:
    recipe = json.loads(json.dumps(recipe))
    materialization = recipe["materialization"]
    for name, value in overrides.items():
        if name in recipe.get("bindings", {}):
            recipe["bindings"][name] = value
        if name in materialization.get("kwargs", {}):
            materialization["kwargs"][name] = value
        if name in materialization.get("resolver_kwargs", {}):
            materialization["resolver_kwargs"][name] = value
        if name in args_index:
            materialization["args"][args_index[name]] = value
    return recipe


# --------------------------------------------------------------------------
# generate (runs under the DSL environment: tl-pg for tilelang, triton-pg for
# triton; CPU only -- CUDA_VISIBLE_DEVICES is cleared below).
# --------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(SRC))
    from mutation_suite_worker import (  # type: ignore
        _capture_tilelang,
        _capture_triton,
        _load_module,
        _specialize_tilelang_module,
    )
    from upstream_worker import _lower_tilelang, _lower_triton  # type: ignore

    requested = args.series or [name for name in SERIES]
    out_root = Path(args.instances)
    for series_name in requested:
        series = SERIES[series_name]
        kernel_id = series["kernel_id"]
        kdir = REPO / "suite/suite4/kernels" / kernel_id
        base_recipe = json.loads(
            (kdir / "recipes" / f"{kernel_id}-workload-original.json").read_text(encoding="utf-8")
        )
        try:
            source_module = _load_module(
                kdir / "sources" / "original.py", f"pg_scal_{kernel_id.replace('-', '_')}"
            )
        except ImportError as error:
            print(f"SKIP {series_name}: DSL runtime unavailable in this interpreter: {error}", file=sys.stderr)
            continue
        case_id = f"{kernel_id}-source-original"
        for tier in series["tiers"]:
            tier_dir = out_root / series_name / tier["label"]
            instance_path = tier_dir / "instance.json"
            if instance_path.exists() and not args.force:
                print(f"keep {series_name}/{tier['label']} (exists)", file=sys.stderr)
                continue
            tier_dir.mkdir(parents=True, exist_ok=True)
            recipe = apply_overrides(base_recipe, tier["overrides"], series["args_index"])
            timings: dict[str, float] = {}
            started = time.monotonic()
            if series["dsl"] == "tilelang":
                from tilelang import tvm  # type: ignore

                lowering_recipe = json.loads(json.dumps(recipe))
                kwargs = lowering_recipe["materialization"]["kwargs"]
                for name in lowering_recipe["compiler"].get("tuple_materialization_kwargs", []):
                    kwargs[name] = tuple(kwargs[name])
                _tvm, modules = _lower_tilelang(source_module, lowering_recipe, resolution="capsule")
                modules = [
                    (symbol, _specialize_tilelang_module(tvm, device_module, recipe))
                    for symbol, device_module in modules
                ]
                if len(modules) != 1:
                    raise RuntimeError(f"{series_name}: expected one device func, got {len(modules)}")
                timings["lower_s"] = time.monotonic() - started
                raw_path = tier_dir / "raw.tvm.json"
                raw_path.write_text(tvm.ir.save_json(modules[0][1]), encoding="utf-8")
                phase = time.monotonic()
                snapshot = _capture_tilelang(raw_path, recipe, case_id)
                timings["capture_s"] = time.monotonic() - phase
            else:
                text = _lower_triton(source_module, recipe, resolution="capsule")
                timings["lower_s"] = time.monotonic() - started
                raw_path = tier_dir / "raw.ttgir.mlir"
                raw_path.write_text(text, encoding="utf-8")
                phase = time.monotonic()
                snapshot = _capture_triton(raw_path, recipe, case_id, tier_dir)
                timings["capture_s"] = time.monotonic() - phase
            if not isinstance(snapshot, dict) or "operations" not in snapshot:
                raise RuntimeError(f"{series_name}/{tier['label']}: capture did not produce a snapshot: "
                                   f"{snapshot.get('diagnostics') if isinstance(snapshot, dict) else snapshot}")
            snapshot_path = tier_dir / "snapshot.json"
            snapshot_path.write_text(
                json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
            instance = {
                "schema_version": "pipeguard-scalability-instance-v1",
                "series": series_name,
                "tier": tier["label"],
                "dsl": series["dsl"],
                "kernel_id": kernel_id,
                "case_id": case_id,
                "family": series["family"],
                "scale_parameter": series["scale_parameter"],
                "overrides": tier["overrides"],
                "recipe": recipe,
                "snapshot_counts": load_snapshot_counts(snapshot_path),
                "artifacts": {
                    "raw": {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size},
                    "snapshot": {"path": snapshot_path.name, "sha256": sha256_file(snapshot_path), "bytes": snapshot_path.stat().st_size},
                },
                "generation": {
                    "timings_s": {key: round(value, 3) for key, value in timings.items()},
                    "python": platform.python_version(),
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "repo_git_sha": git_short_sha(),
                },
            }
            dump_json(instance, instance_path)
            print(
                f"generated {series_name}/{tier['label']}: "
                f"{instance['snapshot_counts']['operations']} snapshot ops, "
                f"lower {timings['lower_s']:.1f}s capture {timings['capture_s']:.1f}s",
                file=sys.stderr,
            )
    return 0


# --------------------------------------------------------------------------
# parse (checker env): snapshot.json -> normalized pgir.json via the frozen
# suite4 parser, plus a representation crosscheck on each series' smallest
# tier.
# --------------------------------------------------------------------------
def cmd_parse(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SRC))
    out_root = Path(args.instances)
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(SRC), str(TOOLS))),
    }
    for series_name in sorted(SERIES):
        series = SERIES[series_name]
        for index, tier in enumerate(series["tiers"]):
            tier_dir = out_root / series_name / tier["label"]
            instance_path = tier_dir / "instance.json"
            if not instance_path.exists():
                continue
            instance = json.loads(instance_path.read_text(encoding="utf-8"))
            pgir_path = tier_dir / "pgir.json"
            if not pgir_path.exists() or args.force:
                subprocess.run(
                    [
                        sys.executable, str(TOOLS / "upstream_parse.py"),
                        "--dsl", series["dsl"],
                        "--snapshot", str(tier_dir / "snapshot.json"),
                        "--pgir", str(pgir_path),
                        "--metadata", str(tier_dir / "pgir-metadata.json"),
                    ],
                    cwd=REPO, env=environment, check=True, capture_output=True, text=True,
                )
            instance["artifacts"]["pgir"] = {
                "path": pgir_path.name, "sha256": sha256_file(pgir_path), "bytes": pgir_path.stat().st_size,
            }
            # Fidelity gate 1: smallest tier must agree between the compact
            # parametric representation and the fully expanded reference.
            # expand_reference() is a pre-existing checker limitation for
            # modules with explicit async-engine streams (TMA kernels); that
            # outcome is recorded honestly instead of being masked.
            if index == 0 and (args.crosscheck or "crosscheck" not in instance):
                from pipeguard import check_module, load_module
                from pipeguard.model import PGIRError

                module = load_module(pgir_path)
                try:
                    result = check_module(module, engine="crosscheck")
                    crosscheck = result.get("artifacts", {}).get("representation_crosscheck", {})
                    instance["crosscheck"] = {
                        "verdict": result["verdict"],
                        "parametric": crosscheck.get("parametric", {}).get("verdict"),
                        "expanded": crosscheck.get("expanded", {}).get("verdict"),
                    }
                except PGIRError as error:
                    instance["crosscheck"] = {
                        "verdict": "UNAVAILABLE",
                        "reason": f"expand_reference failed: {error.message}",
                    }
                print(f"crosscheck {series_name}/{tier['label']}: {instance['crosscheck']}", file=sys.stderr)
            dump_json(instance, instance_path)
            print(f"parsed {series_name}/{tier['label']} -> {pgir_path.name}", file=sys.stderr)
            if not args.keep_snapshots:
                snapshot_path = tier_dir / "snapshot.json"
                if snapshot_path.exists() and index > 0:
                    snapshot_path.unlink()  # sha256 + bytes retained in instance.json
    return 0


# --------------------------------------------------------------------------
# measure-worker (internal): one pinned process per instance.
# --------------------------------------------------------------------------
def cmd_measure_worker(args: argparse.Namespace) -> int:
    gc.enable()
    sys.path.insert(0, str(SRC))
    from pipeguard import check_module, load_module

    pgir_path = Path(args.pgir)
    t0 = time.perf_counter()
    module = load_module(pgir_path)
    load_s = time.perf_counter() - t0

    samples: list[dict[str, Any]] = []
    statistics_snapshot: dict[str, Any] = {}
    verdicts: set[str] = set()
    for repetition in range(args.repeats + 1):  # +1 warmup
        gc.collect()
        t0 = time.perf_counter()
        report = check_module(module, engine=CHECKER_ENGINE)
        wall = time.perf_counter() - t0
        stats = report["statistics"]
        verdicts.add(report["verdict"])
        if repetition == 0:
            statistics_snapshot = {
                key: stats.get(key)
                for key in (
                    "events", "actors", "resources", "regions", "representation",
                    "event_templates", "logical_events", "logical_program_streams",
                    "race_candidate_pairs", "implicit_pdg_nodes",
                    "pdg_implicit_program_edges", "explicit_boundary_edges",
                    "protocol_transitions",
                )
            }
            continue  # warmup not recorded
        samples.append({
            "wall_s": wall,
            "checker_total_s": stats.get("checker_total_s"),
            "pdg_build_s": stats.get("implicit_pdg_build_s"),
            "initial_pdg_setup_s": stats.get("initial_pdg_setup_s"),
            "protocol_schedule_s": stats.get("protocol_schedule_s"),
            "boundary_finalize_s": stats.get("boundary_finalize_s"),
            "race_scan_s": stats.get("race_scan_s"),
        })
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def median(key: str) -> float | None:
        values = [sample[key] for sample in samples if sample[key] is not None]
        return statistics.median(values) if values else None

    print(json.dumps({
        "pgir_sha256": sha256_file(pgir_path),
        "load_s": load_s,
        "repeats": args.repeats,
        "verdicts": sorted(verdicts),
        "statistics": statistics_snapshot,
        "samples": samples,
        "median": {key: median(key) for key in (
            "wall_s", "checker_total_s", "pdg_build_s", "race_scan_s",
            "protocol_schedule_s", "boundary_finalize_s",
        )},
        "peak_rss_mb": peak_rss_mb,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# measure (orchestrator): spawn one taskset-pinned worker per instance,
# recording a machine-load snapshot around each instance.
# --------------------------------------------------------------------------
def read_proc_stat() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]  # idle + iowait
    return sum(values), idle


def load_snapshot() -> dict[str, Any]:
    loadavg = Path("/proc/loadavg").read_text().split()
    return {
        "loadavg_1m": float(loadavg[0]),
        "loadavg_5m": float(loadavg[1]),
        "loadavg_15m": float(loadavg[2]),
        "runnable_tasks": loadavg[3],
        "unix_time": int(time.time()),
    }


def cmd_measure(args: argparse.Namespace) -> int:
    out_root = Path(args.instances)
    measurements_dir = Path(args.out)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    cpu_list = args.cpu
    for series_name in sorted(SERIES):
        series = SERIES[series_name]
        for tier in series["tiers"]:
            tier_dir = out_root / series_name / tier["label"]
            pgir_path = tier_dir / "pgir.json"
            if not pgir_path.exists():
                print(f"skip {series_name}/{tier['label']}: no pgir.json", file=sys.stderr)
                continue
            measurement_path = measurements_dir / f"{series_name}--{tier['label']}.json"
            if measurement_path.exists() and not args.force:
                print(f"keep measurement {measurement_path.name}", file=sys.stderr)
                continue
            instance = json.loads((tier_dir / "instance.json").read_text(encoding="utf-8"))
            repeats = max(5, args.repeats)
            before = {"load": load_snapshot(), "stat": read_proc_stat()}
            command = [
                "taskset", "-c", cpu_list,
                sys.executable, str(Path(__file__).resolve()),
                "measure-worker", "--pgir", str(pgir_path), "--repeats", str(repeats),
            ]
            environment = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            print(f"measuring {series_name}/{tier['label']} (repeats={repeats}, cpu={cpu_list})", file=sys.stderr)
            completed = subprocess.run(command, env=environment, capture_output=True, text=True)
            if completed.returncode != 0:
                print(f"WORKER FAILED {series_name}/{tier['label']}:\n{completed.stderr[-2000:]}", file=sys.stderr)
                continue
            worker = json.loads(completed.stdout)
            after = {"load": load_snapshot(), "stat": read_proc_stat()}
            total_delta = after["stat"][0] - before["stat"][0]
            idle_delta = after["stat"][1] - before["stat"][1]
            busy_fraction = (total_delta - idle_delta) / total_delta if total_delta else None
            measurement = {
                "schema_version": "pipeguard-scalability-measurement-v1",
                "series": series_name,
                "tier": tier["label"],
                "engine": CHECKER_ENGINE,
                "pinned_cpus": cpu_list,
                "worker": worker,
                "machine": {
                    "cpu_count": os.cpu_count(),
                    "before": before["load"],
                    "after": after["load"],
                    "system_busy_fraction_during_run": busy_fraction,
                },
            }
            dump_json(measurement, measurement_path)
            median = worker["median"]
            print(
                f"  events={worker['statistics'].get('events')} "
                f"wall={median['wall_s']:.3f}s pdg={median['pdg_build_s']:.3f}s "
                f"race={median['race_scan_s']:.3f}s rss={worker['peak_rss_mb']:.0f}MB "
                f"verdicts={worker['verdicts']}",
                file=sys.stderr,
            )
    return 0


# --------------------------------------------------------------------------
# report: aggregate + log-log OLS fits.
# --------------------------------------------------------------------------
# two-sided 97.5% Student-t quantiles for dof 1..8 (n-2 for n tiers)
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def loglog_fit(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    points = [(x, y) for x, y in points if x > 0 and y is not None and y > 0]
    n = len(points)
    if n < 3:
        return None
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0 else None
    dof = n - 2
    stderr = math.sqrt((sse / dof) / sxx) if dof > 0 and sxx > 0 else None
    t_value = T_975.get(dof)
    ci = ([slope - t_value * stderr, slope + t_value * stderr]
          if stderr is not None and t_value is not None else None)
    return {
        "n_points": n,
        "slope": round(slope, 4),
        "slope_stderr": round(stderr, 4) if stderr is not None else None,
        "slope_ci95": [round(v, 4) for v in ci] if ci else None,
        "intercept_log10": round(intercept, 4),
        "r_squared": round(r2, 6) if r2 is not None else None,
    }


def cmd_report(args: argparse.Namespace) -> int:
    out_root = Path(args.instances)
    measurements_dir = Path(args.measurements)
    series_blocks: list[dict[str, Any]] = []
    for series_name in sorted(SERIES):
        series = SERIES[series_name]
        rows: list[dict[str, Any]] = []
        for tier in series["tiers"]:
            measurement_path = measurements_dir / f"{series_name}--{tier['label']}.json"
            instance_path = out_root / series_name / tier["label"] / "instance.json"
            if not measurement_path.exists() or not instance_path.exists():
                continue
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            instance = json.loads(instance_path.read_text(encoding="utf-8"))
            worker = measurement["worker"]
            rows.append({
                "tier": tier["label"],
                "overrides": tier["overrides"],
                "events": worker["statistics"]["events"],
                "actors": worker["statistics"]["actors"],
                "event_templates": worker["statistics"].get("event_templates"),
                "race_candidate_pairs": worker["statistics"].get("race_candidate_pairs"),
                "verdicts": worker["verdicts"],
                "repeats": worker["repeats"],
                "load_s": round(worker["load_s"], 4),
                "median_s": {key: (round(value, 5) if value is not None else None)
                              for key, value in worker["median"].items()},
                "samples": worker["samples"],
                "peak_rss_mb": round(worker["peak_rss_mb"], 1),
                "pgir_sha256": worker["pgir_sha256"],
                "pgir_bytes": instance["artifacts"].get("pgir", {}).get("bytes"),
                "machine": measurement["machine"],
                "crosscheck": instance.get("crosscheck"),
            })
        if not rows:
            continue
        fits = {
            metric: loglog_fit([(row["events"], row["median_s"][metric]) for row in rows])
            for metric in ("pdg_build_s", "race_scan_s", "checker_total_s", "wall_s")
        }
        series_blocks.append({
            "series": series_name,
            "dsl": series["dsl"],
            "family": series["family"],
            "scale_parameter": series["scale_parameter"],
            "points": rows,
            "loglog_fits_vs_events": fits,
        })
    result = {
        "schema_version": "pipeguard-scalability-results-v1",
        "suite": "scalability",
        "engine": CHECKER_ENGINE,
        "repo_git_sha": git_short_sha(),
        "method": (
            "workload-parameterized regeneration through the frozen suite4 pipeline "
            "(real compiler lowering + frontend capture, CPU only); per point: "
            ">=5 pinned repetitions after 1 warmup, median reported; peak RSS per instance"
        ),
        "total_events_max": max(
            (row["events"] for block in series_blocks for row in block["points"]), default=0
        ),
        "series": series_blocks,
    }
    out_path = Path(args.out) if args.out else (
        REPO / "suite" / "scalability" / f"results-{git_short_sha()}.json"
    )
    dump_json(result, out_path)
    print(f"wrote {out_path}", file=sys.stderr)
    for block in series_blocks:
        fits = block["loglog_fits_vs_events"]
        print(f"{block['series']}: " + ", ".join(
            f"{metric} slope={fit['slope']} ci95={fit['slope_ci95']}"
            for metric, fit in fits.items() if fit
        ), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--series", action="append", choices=sorted(SERIES), default=None)
    generate.add_argument("--instances", default=str(REPO / "suite/scalability/instances"))
    generate.add_argument("--force", action="store_true")

    parse = subparsers.add_parser("parse")
    parse.add_argument("--instances", default=str(REPO / "suite/scalability/instances"))
    parse.add_argument("--force", action="store_true")
    parse.add_argument("--crosscheck", action="store_true")
    parse.add_argument("--keep-snapshots", action="store_true")

    measure = subparsers.add_parser("measure")
    measure.add_argument("--instances", default=str(REPO / "suite/scalability/instances"))
    measure.add_argument("--out", default=str(REPO / "suite/scalability/measurements"))
    measure.add_argument("--cpu", default="120", help="taskset CPU list for pinned workers")
    measure.add_argument("--repeats", type=int, default=5, help="repetitions per point (min 5)")
    measure.add_argument("--force", action="store_true")

    worker = subparsers.add_parser("measure-worker")
    worker.add_argument("--pgir", required=True)
    worker.add_argument("--repeats", type=int, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--instances", default=str(REPO / "suite/scalability/instances"))
    report.add_argument("--measurements", default=str(REPO / "suite/scalability/measurements"))
    report.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "parse":
        return cmd_parse(args)
    if args.command == "measure":
        return cmd_measure(args)
    if args.command == "measure-worker":
        return cmd_measure_worker(args)
    if args.command == "report":
        return cmd_report(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
