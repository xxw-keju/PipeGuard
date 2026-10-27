from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
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


def _fingerprint(repository_root: Path, catalog_path: Path, variant: dict[str, Any], workload: dict[str, Any], worker: Path) -> str:
    source_path = repository_root / variant["source"]["path"]
    recipe_path = repository_root / workload["recipe"]["path"]
    material = {
        "version": "pipeguard-mutation-runner-v1",
        "catalog_sha256": sha256_file(catalog_path),
        "variant": variant,
        "workload": workload,
        "source_sha256": sha256_file(source_path),
        "recipe_sha256": sha256_file(recipe_path),
        "worker_sha256": sha256_file(worker),
    }
    return sha256_text(canonical_json(material))


def _environment(repository_root: Path, case_directory: Path, dsl: str, triton_ptxas: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "HOME": str(case_directory / "home"),
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
        "PYTHONPATH": os.pathsep.join((str(repository_root / "src"), str(repository_root / "tools"))),
    }
    if dsl == "triton":
        environment["TRITON_CACHE_DIR"] = str(case_directory / "triton-cache")
        environment["TRITON_PTXAS_PATH"] = str(triton_ptxas)
        environment["TRITON_KERNEL_OVERRIDE"] = "0"
        environment["TRITON_STORE_BINARY_ONLY"] = "0"
    return environment


def _merge_numeric(case_directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    numeric_path = case_directory / "numeric.json"
    if not numeric_path.exists():
        return result
    numeric = load_json(numeric_path)
    if numeric.get("case_id") != result.get("case_id"):
        raise ValueError(f"numeric result belongs to another case: {numeric_path}")
    result["numeric_status"] = numeric["numeric_status"]
    result["numeric_metrics"] = numeric.get("metrics", {})
    existing_codes = {item["code"] for item in result.get("diagnostics", [])}
    result.setdefault("diagnostics", []).extend(
        item for item in numeric.get("diagnostics", []) if item["code"] not in existing_codes
    )
    result.setdefault("artifacts", {})["numeric"] = {
        "path": numeric_path.name,
        "sha256": sha256_file(numeric_path),
    }
    return result


def _timeout_result(request: dict[str, Any], timeout: float) -> dict[str, Any]:
    case = request["case"]
    return {
        "schema_version": "pipeguard-mutation-run-result-v1",
        "fingerprint": request["fingerprint"],
        "case_id": case["case_id"],
        "kernel_id": case["kernel_id"],
        "variant_id": case["variant_id"],
        "spec_id": case["spec_id"],
        "dsl": request["dsl"],
        "pipeline_status": "TIMEOUT",
        "checker_status": "TIMEOUT",
        "numeric_status": "NOT_RUN",
        "diagnostics": [{"code": "SUITE4_WORKER_TIMEOUT", "message": f"worker exceeded {timeout} seconds"}],
        "artifacts": {},
        "timings": {"timeout": timeout},
    }


def _run_one(item: dict[str, Any], arguments: argparse.Namespace, output: Path) -> tuple[dict[str, Any], bool]:
    case, kernel, variant, workload = (item[key] for key in ("case", "kernel", "variant", "workload"))
    fingerprint = _fingerprint(arguments.repository_root, arguments.catalog, variant, workload, arguments.worker)
    case_directory = output / safe_component(case["case_id"])
    result_path = case_directory / "result.json"
    if result_path.exists() and not arguments.force:
        result = load_json(result_path)
        if result.get("fingerprint") == fingerprint:
            result = _merge_numeric(case_directory, result)
            atomic_write(result_path, pretty_json(result))
            return result, True
    case_directory.mkdir(parents=True, exist_ok=True)
    recipe = load_json(arguments.repository_root / workload["recipe"]["path"])
    request = {
        "fingerprint": fingerprint,
        "dsl": kernel["dsl"],
        "source_path": str((arguments.repository_root / variant["source"]["path"]).resolve(strict=True)),
        "recipe": recipe,
        "case": case,
    }
    request_path = case_directory / "request.json"
    atomic_write(request_path, pretty_json(request))
    interpreter = arguments.tilelang_python if kernel["dsl"] == "tilelang" else arguments.triton_python
    process = subprocess.Popen(
        [str(interpreter), str(arguments.worker), "--request", str(request_path), "--output", str(case_directory)],
        cwd=arguments.repository_root,
        env=_environment(arguments.repository_root, case_directory, kernel["dsl"], arguments.triton_ptxas),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=arguments.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        result = _timeout_result(request, arguments.timeout)
        atomic_write(result_path, pretty_json(result))
        return result, False
    atomic_write(case_directory / "worker.stdout", stdout[-1_000_000:])
    atomic_write(case_directory / "worker.stderr", stderr[-1_000_000:])
    if result_path.exists():
        result = load_json(result_path)
    else:
        result = _timeout_result(request, arguments.timeout)
        result["pipeline_status"] = "INFRA_ERROR"
        result["checker_status"] = "INFRA_ERROR"
        result["diagnostics"] = [{"code": "SUITE4_WORKER_RESULT_MISSING", "message": stderr[-4000:] or stdout[-4000:] or f"worker exited {process.returncode}"}]
        atomic_write(result_path, pretty_json(result))
    result = _merge_numeric(case_directory, result)
    atomic_write(result_path, pretty_json(result))
    return result, False


def _selected(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    _lock, catalog, _baseline, _summary = validate_mutation_suite(arguments.repository_root, catalog_path=arguments.catalog)
    kernels = {item["kernel_id"]: item for item in catalog["kernels"]}
    variants = {item["variant_id"]: item for item in catalog["variants"]}
    workloads = {item["spec_id"]: item for item in catalog["workloads"]}
    kernel_filter = set(arguments.kernel)
    variant_filter = set(arguments.variant)
    spec_filter = set(arguments.spec)
    selected = []
    for case in catalog["cases"]:
        kernel = kernels[case["kernel_id"]]
        variant = variants[case["variant_id"]]
        workload = workloads[case["spec_id"]]
        if kernel_filter and kernel["kernel_id"] not in kernel_filter:
            continue
        if variant_filter and variant["variant_id"] not in variant_filter:
            continue
        if spec_filter and workload["spec_id"] not in spec_filter:
            continue
        if arguments.original_sources_only and variant["role"] != "ORIGINAL":
            continue
        if arguments.original_workloads_only and workload["role"] != "ORIGINAL":
            continue
        selected.append({"case": case, "kernel": kernel, "variant": variant, "workload": workload})
    if not selected:
        raise ValueError("Suite 4 filters selected no cases")
    return selected


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Suite 4 native lowering, capture, and checking")
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("mutation_suite_worker.py"))
    parser.add_argument("--tilelang-python", type=Path, default=DEFAULT_TILELANG_PYTHON)
    parser.add_argument("--triton-python", type=Path, default=DEFAULT_TRITON_PYTHON)
    parser.add_argument("--triton-ptxas", type=Path, default=DEFAULT_TRITON_PTXAS)
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--original-sources-only", action="store_true")
    parser.add_argument("--original-workloads-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.repository_root = args.repository_root.resolve(strict=True)
    args.catalog = (args.catalog or args.repository_root / "corpus/mutations/suite4/catalog.json").resolve(strict=True)
    args.output = args.output.resolve()
    args.worker = args.worker.resolve(strict=True)
    args.tilelang_python = args.tilelang_python.resolve(strict=True)
    args.triton_python = args.triton_python.resolve(strict=True)
    args.triton_ptxas = args.triton_ptxas.resolve(strict=True)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    arguments.output.mkdir(parents=True, exist_ok=True)
    selected = _selected(arguments)
    completed: list[dict[str, Any]] = []
    resumed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(arguments.jobs, len(selected))) as executor:
        futures = [executor.submit(_run_one, item, arguments, arguments.output) for item in selected]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result, was_resumed = future.result()
            completed.append(result)
            resumed += int(was_resumed)
            print(canonical_json({"completed": index, "total": len(selected), "case_id": result["case_id"], "pipeline_status": result["pipeline_status"], "checker_status": result["checker_status"], "resumed": was_resumed}), flush=True)
    summary = {
        "schema_version": "pipeguard-mutation-run-summary-v1",
        "selected": len(selected),
        "completed": len(completed),
        "resumed": resumed,
        "pipeline_statuses": dict(sorted(Counter(item["pipeline_status"] for item in completed).items())),
        "checker_statuses": dict(sorted(Counter(item["checker_status"] for item in completed).items())),
    }
    atomic_write(arguments.output / "summary.json", pretty_json(summary))
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
