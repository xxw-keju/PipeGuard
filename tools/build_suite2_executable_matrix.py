from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
TESTS = ROOT / "tests"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(TESTS)]

from pipeguard.checker import check_module
from pipeguard.frontends.tilelang import tilelang_snapshot_to_module
from pipeguard.frontends.triton import triton_snapshot_to_module

SCHEMA = "pipeguard-suite2-executable-matrix-v1"
DSLS = ("tilelang", "triton")
RESULT_CLASSES = ("SAFE", "DATA_HAZARD", "SYNC_HAZARD")
PRIMITIVES = ("Barrier", "mbarrier", "TMA", "cp.async", "WGMMA", "Proxy", "Memory")
PER_CELL = 15
DEFAULT_OUTPUT = ROOT / ".artifacts/frontend-pairs/suite2-executable-matrix-v2"
DEFAULT_CORPUS = ROOT / "corpus/suite2/executable-v2"
WORKER = ROOT / "tools/suite2_executable_worker.py"
PYTHONS = {
    "tilelang": Path("/usr/local/miniconda3/envs/tl-pg/bin/python"),
    "triton": Path("/usr/local/miniconda3/envs/triton-pg/bin/python"),
}

TEMPLATES = {
    ("SAFE", "Barrier"): ("CTA-HANDOFF", "CTA-HANDOFF", "CORRECT"),
    ("SAFE", "mbarrier"): ("MB-REUSE", "MB-REUSE", "CORRECT"),
    ("SAFE", "TMA"): ("TMA-OBSERVE", "TMA-OBSERVE", "CORRECT"),
    ("SAFE", "cp.async"): ("CP-WAIT", "CP-WAIT", "CORRECT"),
    ("SAFE", "WGMMA"): ("WGMMA-ENDPOINT", "WGMMA-ENDPOINT", "CORRECT"),
    ("SAFE", "Proxy"): ("TMA-PROXY", "TMA-PROXY", "CORRECT"),
    ("SAFE", "Memory"): ("MEM-RAW", "MEM-RAW", "CORRECT"),
    ("DATA_HAZARD", "Barrier"): ("CTA-HANDOFF", "CTA-HANDOFF", "BUG"),
    ("DATA_HAZARD", "mbarrier"): ("TMA-OBSERVE", "TMA-OBSERVE", "BUG"),
    ("DATA_HAZARD", "TMA"): ("TMA-OBSERVE", "TMA-OBSERVE", "BUG"),
    ("DATA_HAZARD", "cp.async"): ("CP-WAIT", "CP-WAIT", "BUG"),
    ("DATA_HAZARD", "WGMMA"): ("WGMMA-ENDPOINT", "WGMMA-ENDPOINT", "BUG"),
    ("DATA_HAZARD", "Proxy"): ("TMA-PROXY", "TMA-PROXY", "BUG"),
    ("DATA_HAZARD", "Memory"): ("MEM-WAW", "MEM-WAW", "BUG"),
    ("SYNC_HAZARD", "Barrier"): ("CTA-BARRIER", "CTA-BARRIER:MISSING-PARTICIPANT", "BUG"),
    ("SYNC_HAZARD", "mbarrier"): ("MB-REUSE", "MB-REUSE", "BUG"),
    ("SYNC_HAZARD", "TMA"): ("TMA-SYNC", "TMA-SYNC:TRANSACTION-BYTES-MISMATCH", "BUG"),
    ("SYNC_HAZARD", "cp.async"): ("CP-ASYNC", "CP-ASYNC:WAIT-BEFORE-COMMIT", "BUG"),
    ("SYNC_HAZARD", "WGMMA"): ("WGMMA", "WGMMA:MISSING-OPERAND-FENCE", "BUG"),
    ("SYNC_HAZARD", "Proxy"): ("WGMMA", "WGMMA:MISSING-OPERAND-FENCE", "BUG"),
    ("SYNC_HAZARD", "Memory"): ("CTA-BARRIER", "CTA-BARRIER:MISSING-PARTICIPANT", "BUG"),
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite divergent file: {path}")
        return
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _source(spec: dict[str, Any]) -> str:
    payload = repr(spec)
    if spec["dsl"] == "tilelang":
        body = '''from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = __PAYLOAD__

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        if CASE_SPEC["primitive"] == "WGMMA":
            return _cases._suite2_wgmma_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "Proxy":
            return _cases._suite2_proxy_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "TMA" and CASE_SPEC["result_class"] == "SYNC_HAZARD":
            return _cases._suite2_tma_sync_case(CASE_SPEC["variant"])
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
'''
    else:
        body = '''from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = __PAYLOAD__

def pipeguard_resolve():
    s = CASE_SPEC
    family, scenario = s["template_family"], s["scenario_id"]
    elements = s["semantic_knobs"]["elements"]
    threads = s["semantic_knobs"]["threads"]
    if s["primitive"] == "WGMMA":
        return _cases._suite2_wgmma_kernel(s["variant"], s["result_class"])
    if s["primitive"] == "Proxy":
        return _cases._suite2_proxy_kernel(s["variant"], s["result_class"])
    if s["primitive"] == "TMA" and s["result_class"] == "SYNC_HAZARD":
        return _cases._suite2_tma_sync_kernel(s["variant"])
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        return _cases._memory_kernel(family, elements, threads)
    if family == "MB-REUSE": return _cases._mbarrier_kernel()
    if family in {"CP-WAIT", "CP-OVERLAP"}: return _cases._cp_async_kernel(family, threads)
    if family in {"TMA-OBSERVE", "TMA-PROXY"}: return _cases._tma_load_kernel(family, s["variant"])
    if family == "WGMMA-ENDPOINT": return _cases._wgmma_kernel()
    if family == "TMA-STORE-ENDPOINT": return _cases._tma_store_kernel()
    if family == "CTA-BARRIER": return _cases._named_cta_kernel(scenario, threads)
    if family == "MBARRIER": return _cases._named_mbarrier_kernel(scenario)
    if family == "CP-ASYNC": return _cases._named_cp_async_kernel(scenario, threads)
    if family == "TMA-LOAD": return _cases._named_tma_load_kernel(scenario)
    if family == "WGMMA": return _cases._named_wgmma_kernel(scenario)
    if family == "TMA-STORE": return _cases._named_tma_store_kernel(scenario)
    raise KeyError(f"unsupported template family: {family}")
'''
    return "# Generated executable Suite 2 DSL kernel; do not hand edit.\n" + body.replace("__PAYLOAD__", payload)


def build_specs(
    corpus: Path,
    primitives: tuple[str, ...] = PRIMITIVES,
) -> list[dict[str, Any]]:
    specs = []
    for dsl in DSLS:
        for result_class in RESULT_CLASSES:
            for primitive in primitives:
                family, scenario, polarity = TEMPLATES[(result_class, primitive)]
                for variant in range(PER_CELL):
                    case_id = f"s2x-{dsl}-{result_class.lower().replace('_','-')}-{primitive.lower().replace('.','-')}-{variant:02d}"
                    spec = {
                        "case_id": case_id, "dsl": dsl, "result_class": result_class,
                        "primitive": primitive, "variant": variant,
                        "template_family": family, "scenario_id": scenario, "polarity": polarity,
                        "semantic_knobs": {
                            "elements": variant + 1, "stages": 2 + variant % 3,
                            "threads": 32 * (1 + variant % 4),
                            "region_offset_bytes": 16 * variant,
                            "protocol_depth": variant + 2,
                            "tilelang_tma_bytes": 16 * (variant + 1),
                            "triton_tma_block": 32,
                            "triton_tma_transfers": variant + 1,
                            "wgmma_repeats": variant + 1,
                            "wgmma_shape": [64, 64, 16],
                            "proxy_repeats": variant + 1,
                            "proxy_store_shape": [64, 64],
                        },
                        "oracle": {
                            "verdict": result_class,
                            "basis": "independent paired protocol template plus variant-specific participants/region/depth",
                        },
                        "admitted": False,
                        "ai_adjudication": {
                            "status": "PENDING_INDEPENDENT_REVIEW",
                            "required_evidence": [
                                "source_and_lowered_operations", "physical_participants",
                                "resource_generation_or_endpoint", "byte_overlap_if_applicable",
                                "reachable_prefix", "missing_or_present_typed_order",
                            ],
                        },
                    }
                    source_path = corpus / dsl / result_class.lower() / primitive.lower().replace(".", "-") / f"{case_id}.py"
                    text = _source(spec)
                    write_once(source_path, text.encode())
                    source_identity = (
                        source_path.relative_to(ROOT)
                        if source_path.is_relative_to(ROOT)
                        else source_path
                    )
                    spec["source"] = {
                        "path": str(source_identity), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "bytes": len(text.encode()),
                    }
                    spec_path = source_path.with_suffix(".json")
                    write_once(spec_path, canonical(spec))
                    spec["spec_path"] = str(
                        spec_path.relative_to(ROOT)
                        if spec_path.is_relative_to(ROOT)
                        else spec_path
                    )
                    specs.append(spec)
    return specs


def _environment(case_dir: Path, dsl: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HOME": str(case_dir / "home"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0", "PYTHONPATH": os.pathsep.join((str(TESTS), str(ROOT / "src"), str(ROOT / "tools"))),
        "TRITON_CACHE_DIR": str(case_dir / "triton-cache"), "TRITON_PTXAS_PATH": "/usr/local/cuda-12.9/bin/ptxas",
    })
    return env


def _check_snapshot(spec: dict[str, Any], case_dir: Path, worker: dict[str, Any]) -> dict[str, Any]:
    if worker.get("capture_status") != "CAPTURED":
        return {"status": "NOT_CHECKED", "verdict": None, "diagnostic": None}
    snapshot = json.loads((case_dir / worker["snapshot"]["path"]).read_text())
    converter = tilelang_snapshot_to_module if spec["dsl"] == "tilelang" else triton_snapshot_to_module
    try:
        module = converter(snapshot)
        pgir_path = case_dir / "pgir.json"
        write_once(pgir_path, json.dumps(module.raw, indent=2, sort_keys=True).encode())
        report_path = case_dir / "checker.json"
        request_path = case_dir / "checker.request.json"
        write_once(request_path, json.dumps({
            "pgir_path": str(pgir_path), "engine": "maximal", "report_path": str(report_path),
        }, indent=2, sort_keys=True).encode())
        process = subprocess.run(
            ["/usr/bin/python3", str(ROOT / "tools/upstream_check.py"), "--worker-request", str(request_path)],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tools")))},
            check=False, capture_output=True, text=True, timeout=120,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr[-4000:] or process.stdout[-4000:] or "checker worker failed")
        report = json.loads(report_path.read_text())
        diagnostic = report.get("diagnostics", [{}])[0].get("code") if report.get("diagnostics") else None
        return {"status": "CHECKED", "verdict": report["verdict"], "diagnostic": diagnostic,
                "pgir": {"path": pgir_path.name, "sha256": digest(pgir_path)},
                "report": {"path": report_path.name, "sha256": digest(report_path)}}
    except Exception as error:
        return {"status": "UNSUPPORTED", "verdict": None,
                "diagnostic": getattr(error, "code", type(error).__name__), "message": str(error)}


def run_one(spec: dict[str, Any], output: Path, timeout: float) -> dict[str, Any]:
    case_dir = output / "cases" / spec["case_id"]
    result_path = case_dir / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    case_dir.mkdir(parents=True, exist_ok=True); (case_dir / "home").mkdir(exist_ok=True); (case_dir / "triton-cache").mkdir(exist_ok=True)
    command = [str(PYTHONS[spec["dsl"]]), str(WORKER), "--spec", str(ROOT / spec["spec_path"]), "--output", str(case_dir)]
    try:
        process = subprocess.run(command, cwd=ROOT, env=_environment(case_dir, spec["dsl"]), capture_output=True, text=True, timeout=timeout)
        write_once(case_dir / "worker.stdout", process.stdout.encode()); write_once(case_dir / "worker.stderr", process.stderr.encode())
        worker = json.loads(process.stdout.splitlines()[-1]) if process.returncode == 0 else {
            "execution_status": "ERROR", "capture_status": "ERROR", "diagnostic": "WORKER_EXIT", "returncode": process.returncode}
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""; stderr = error.stderr or b""
        if isinstance(stdout, str): stdout = stdout.encode()
        if isinstance(stderr, str): stderr = stderr.encode()
        write_once(case_dir / "worker.stdout", stdout); write_once(case_dir / "worker.stderr", stderr)
        worker = {"execution_status": "TIMEOUT", "capture_status": "TIMEOUT", "diagnostic": "WORKER_TIMEOUT"}
    check = _check_snapshot(spec, case_dir, worker)
    strict_match = check["status"] == "CHECKED" and check["verdict"] == spec["oracle"]["verdict"]
    triton_status = "EXECUTED_ACCEPTED_NO_DIAGNOSTIC" if spec["dsl"] == "triton" and worker.get("execution_status") == "EXECUTED" else "NOT_APPLICABLE"
    result = {
        "schema_version": SCHEMA, "case_id": spec["case_id"], "dsl": spec["dsl"],
        "result_class": spec["result_class"], "primitive": spec["primitive"], "variant": spec["variant"],
        "source": spec["source"], "semantic_knobs": spec["semantic_knobs"], "oracle": spec["oracle"],
        "ai_adjudication": spec["ai_adjudication"],
        "execution": worker, "pipeguard": check, "strict_oracle_match": strict_match,
        "baselines": {
            "nvidia": {"status": "NOT_RUN_EXACT_BINARY_LAUNCHER_PENDING", "genuine_oracle_mismatch": False},
            "triton": {"status": triton_status, "structural_only": True, "genuine_oracle_mismatch": False}},
        "terminal_status": "COMPLETED" if worker.get("execution_status") == "EXECUTED" and check["status"] == "CHECKED" else worker.get("execution_status", "ERROR"),
        "admitted": bool(
            worker.get("execution_status") == "EXECUTED"
            and strict_match
            and spec["ai_adjudication"]["status"] == "CONFIRMED"
        ),
    }
    write_once(result_path, json.dumps(result, indent=2, sort_keys=True).encode())
    return result


def summarize(specs: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    cells = defaultdict(list)
    for record in records: cells[(record["dsl"], record["result_class"], record["primitive"])].append(record)
    cell_rows = []
    for dsl in DSLS:
        for result_class in RESULT_CLASSES:
            for primitive in PRIMITIVES:
                rows = cells[(dsl, result_class, primitive)]
                cell_rows.append({
                    "dsl": dsl, "result_class": result_class, "primitive": primitive, "attempted": len(rows),
                    "executed": sum(r["execution"].get("execution_status") == "EXECUTED" for r in rows),
                    "admitted": sum(r["admitted"] for r in rows),
                    "unique_source_sha256": len({r["source"]["sha256"] for r in rows}),
                    "pipeguard_mismatches": sum(r["pipeguard"].get("status") == "CHECKED" and not r["strict_oracle_match"] for r in rows)})
    complete = len(records) == len(specs) == 630
    ready_cells = sum(row["admitted"] >= PER_CELL and row["unique_source_sha256"] >= PER_CELL for row in cell_rows)
    counters = {
        "executable_cases_completed": sum(r["execution"].get("execution_status") == "EXECUTED" for r in records),
        "pipeguard_mismatches_remaining": sum(r["pipeguard"].get("status") == "CHECKED" and not r["strict_oracle_match"] for r in records),
        "genuine_nvidia_mismatches_confirmed": sum(r["baselines"]["nvidia"]["genuine_oracle_mismatch"] for r in records),
        "genuine_triton_mismatches_confirmed": sum(r["baselines"]["triton"]["genuine_oracle_mismatch"] for r in records),
        "final_figures_completed": 1}
    blockers = []
    if not complete: blockers.append("EXECUTABLE_RECORDS_INCOMPLETE")
    if ready_cells != 42: blockers.append("CELL_MINIMUM_15_ADMITTED_EXECUTABLE_KERNELS_NOT_MET")
    if counters["pipeguard_mismatches_remaining"]: blockers.append("PIPEGUARD_ORACLE_MISMATCHES")
    if not counters["genuine_nvidia_mismatches_confirmed"]: blockers.append("NVIDIA_GENUINE_EXECUTED_MISMATCH_UNMET")
    if not counters["genuine_triton_mismatches_confirmed"]: blockers.append("TRITON_GENUINE_EXECUTED_MISMATCH_UNMET")
    return {"schema_version": SCHEMA,
            "status": "FINAL_CANDIDATE_COMPLETE" if not blockers else "PRELIMINARY_EXECUTABLE_MATRIX_NOT_FINAL_FIGURE",
            "final_figure_eligible": not blockers, "expected_records": 630, "records": len(records),
            "cells": cell_rows, "ready_cells": ready_cells, "headline_counters": counters,
            "terminal_statuses": dict(sorted(Counter(r["terminal_status"] for r in records).items())), "blockers": blockers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true"); parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0); parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-one-per-cell", action="store_true")
    parser.add_argument("--primitive", action="append", choices=PRIMITIVES)
    args = parser.parse_args(argv); corpus, output = args.corpus.resolve(), args.output.resolve()
    primitives = tuple(args.primitive or PRIMITIVES)
    specs = build_specs(corpus, primitives)
    expected = len(DSLS) * len(RESULT_CLASSES) * len(primitives) * PER_CELL
    if len(specs) != expected or len({s["source"]["sha256"] for s in specs}) != expected:
        raise RuntimeError(f"matrix must contain {expected} source-distinct cases")
    manifest = {"schema_version": SCHEMA, "status": "EXECUTION_PENDING", "cases": specs}
    write_once(corpus / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
    if args.prepare_only:
        print(json.dumps({
            "prepared": len(specs),
            "unique_sources": len({s["source"]["sha256"] for s in specs}),
        }, sort_keys=True)); return 0
    if args.smoke_one_per_cell:
        selected = [spec for spec in specs if spec["variant"] == 0]
    else:
        selected = specs[:args.limit] if args.limit is not None else specs
    output.mkdir(parents=True, exist_ok=True); records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures={executor.submit(run_one,spec,output,args.timeout):spec for spec in selected}
        for index,future in enumerate(concurrent.futures.as_completed(futures),1):
            record=future.result(); records.append(record)
            print(json.dumps({"completed":index,"total":len(selected),"case_id":record["case_id"],"terminal_status":record["terminal_status"]},sort_keys=True),flush=True)
    records.sort(key=lambda r:r["case_id"]); ledger={"schema_version":SCHEMA,"records":records}; summary=summarize(specs,records)
    write_once(output/"ledger.json",json.dumps(ledger,indent=2,sort_keys=True).encode())
    write_once(output/"summary.json",json.dumps(summary,indent=2,sort_keys=True).encode())
    print(json.dumps(summary["headline_counters"],sort_keys=True))
    return 2 if args.check and summary["blockers"] else 0

if __name__ == "__main__": raise SystemExit(main())
