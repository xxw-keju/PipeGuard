from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mutation_suite import load_json, sha256_file, validate_mutation_suite  # noqa: E402
from upstream_common import atomic_write, canonical_json, pretty_json, safe_component, sha256_text  # noqa: E402


DEFAULT_TILELANG_PYTHON = Path("/usr/local/miniconda3/envs/tl-pg/bin/python")
DEFAULT_TRITON_PYTHON = Path("/usr/local/miniconda3/envs/triton-pg/bin/python")
DEFAULT_TRITON_PTXAS = Path("/usr/local/cuda-12.9/bin/ptxas")


def _environment(repository_root: Path, case_directory: Path, dsl: str, ptxas: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "HOME": str(case_directory / "numeric-home"),
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(repository_root / "src"), str(repository_root / "tools"))),
        "TZ": "UTC",
    }
    if dsl == "triton":
        env["TRITON_CACHE_DIR"] = str(case_directory / "numeric-triton-cache")
        env["TRITON_PTXAS_PATH"] = str(ptxas)
    return env


def _selected(arguments: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    _lock, catalog, _baseline, _summary = validate_mutation_suite(arguments.repository_root)
    kernels = {item["kernel_id"]: item for item in catalog["kernels"]}
    variants = {item["variant_id"]: item for item in catalog["variants"]}
    workloads = {item["spec_id"]: item for item in catalog["workloads"]}
    selected = []
    for case in catalog["cases"]:
        kernel, variant, workload = kernels[case["kernel_id"]], variants[case["variant_id"]], workloads[case["spec_id"]]
        if arguments.kernel and kernel["kernel_id"] not in arguments.kernel:
            continue
        if arguments.variant and variant["variant_id"] not in arguments.variant:
            continue
        if arguments.spec and workload["spec_id"] not in arguments.spec:
            continue
        if arguments.original_sources_only and variant["role"] != "ORIGINAL":
            continue
        if arguments.original_workloads_only and workload["role"] != "ORIGINAL":
            continue
        selected.append((case, kernel, variant, workload))
    if not selected:
        raise ValueError("Suite 4 numeric filters selected no cases")
    return selected


def _merge_pipeline_result(case_directory: Path, numeric: dict[str, Any]) -> None:
    result_path = case_directory / "result.json"
    if result_path.exists():
        result = load_json(result_path)
    else:
        result = {
            "schema_version": "pipeguard-mutation-run-result-v1",
            "case_id": numeric["case_id"],
            "pipeline_status": "UNASSESSED",
            "checker_status": "NOT_RUN",
            "diagnostics": [],
            "artifacts": {},
            "timings": {},
        }
    result["numeric_status"] = numeric["numeric_status"]
    result["numeric_metrics"] = numeric.get("metrics", {})
    result["diagnostics"].extend(numeric.get("diagnostics", []))
    numeric_path = case_directory / "numeric.json"
    if numeric_path.exists():
        result["artifacts"]["numeric"] = {"path": "numeric.json", "sha256": sha256_file(numeric_path)}
    atomic_write(result_path, pretty_json(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Suite 4 numeric validation")
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("mutation_suite_numeric_worker.py"))
    parser.add_argument("--tilelang-python", type=Path, default=DEFAULT_TILELANG_PYTHON)
    parser.add_argument("--triton-python", type=Path, default=DEFAULT_TRITON_PYTHON)
    parser.add_argument("--triton-ptxas", type=Path, default=DEFAULT_TRITON_PTXAS)
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--original-sources-only", action="store_true")
    parser.add_argument("--original-workloads-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.repository_root = args.repository_root.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.worker = args.worker.resolve(strict=True)
    args.tilelang_python = args.tilelang_python.resolve(strict=True)
    args.triton_python = args.triton_python.resolve(strict=True)
    args.triton_ptxas = args.triton_ptxas.resolve(strict=True)

    statuses: Counter[str] = Counter()
    selected = _selected(args)
    for index, (case, kernel, variant, workload) in enumerate(selected, 1):
        case_directory = args.output / safe_component(case["case_id"])
        case_directory.mkdir(parents=True, exist_ok=True)
        recipe = load_json(args.repository_root / workload["recipe"]["path"])
        fingerprint = sha256_text(canonical_json({
            "version": "pipeguard-mutation-numeric-v1",
            "variant_id": variant["variant_id"],
            "source_sha256": variant["source"]["sha256"],
            "recipe_sha256": workload["recipe"]["sha256"],
            "worker_sha256": sha256_file(args.worker),
        }))
        numeric_path = case_directory / "numeric.json"
        if numeric_path.exists() and not args.force:
            numeric = load_json(numeric_path)
            if numeric.get("fingerprint") == fingerprint:
                statuses[numeric["numeric_status"]] += 1
                _merge_pipeline_result(case_directory, numeric)
                print(canonical_json({"completed": index, "total": len(selected), "case_id": case["case_id"], "numeric_status": numeric["numeric_status"], "resumed": True}), flush=True)
                continue
        request = {
            "fingerprint": fingerprint,
            "case_id": case["case_id"],
            "kernel_id": kernel["kernel_id"],
            "variant_id": variant["variant_id"],
            "dsl": kernel["dsl"],
            "source_path": str((args.repository_root / variant["source"]["path"]).resolve(strict=True)),
            "recipe": recipe,
        }
        request_path = case_directory / "numeric-request.json"
        atomic_write(request_path, pretty_json(request))
        interpreter = args.tilelang_python if kernel["dsl"] == "tilelang" else args.triton_python
        process = subprocess.Popen(
            [str(interpreter), str(args.worker), "--request", str(request_path), "--output", str(case_directory)],
            cwd=args.repository_root,
            env=_environment(args.repository_root, case_directory, kernel["dsl"], args.triton_ptxas),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timeout = recipe["numeric"]["timeout_seconds"]
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            phase_path = case_directory / "numeric-phase.json"
            phase = load_json(phase_path).get("phase") if phase_path.exists() else "initialization"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            numeric = {
                "schema_version": "pipeguard-mutation-numeric-result-v1",
                "fingerprint": fingerprint,
                "case_id": case["case_id"],
                "numeric_status": "HANG" if phase == "launch" else "COMPILE_ERROR",
                "diagnostics": [{"code": "SUITE4_NUMERIC_TIMEOUT", "message": f"numeric worker exceeded {timeout} seconds", "details": {"phase": phase}}],
                "metrics": {},
            }
            atomic_write(numeric_path, pretty_json(numeric))
            stdout, stderr = "", ""
        else:
            if numeric_path.exists():
                numeric = load_json(numeric_path)
            else:
                numeric = {
                    "schema_version": "pipeguard-mutation-numeric-result-v1",
                    "fingerprint": fingerprint,
                    "case_id": case["case_id"],
                    "numeric_status": "INFRA_ERROR",
                    "diagnostics": [{"code": "SUITE4_NUMERIC_RESULT_MISSING", "message": stderr[-4000:] or stdout[-4000:] or f"worker exited {process.returncode}"}],
                    "metrics": {},
                }
                atomic_write(numeric_path, pretty_json(numeric))
        atomic_write(case_directory / "numeric.stdout", stdout[-1_000_000:])
        atomic_write(case_directory / "numeric.stderr", stderr[-1_000_000:])
        _merge_pipeline_result(case_directory, numeric)
        statuses[numeric["numeric_status"]] += 1
        print(canonical_json({"completed": index, "total": len(selected), "case_id": case["case_id"], "numeric_status": numeric["numeric_status"], "resumed": False}), flush=True)
    print(canonical_json({"selected": len(selected), "numeric_statuses": dict(sorted(statuses.items()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
