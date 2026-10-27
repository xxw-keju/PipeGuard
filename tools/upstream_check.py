#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from upstream_common import atomic_write, canonical_json, git_commit, pretty_json, safe_component, sha256_bytes, sha256_text
from upstream_suite import CHECKER_RESULT_VERSION, validate_checker_result, validate_suite


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _relative_file(root: Path, text: str) -> Path:
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path escapes case directory: {text}")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {text}")
    return path


def _checker_fingerprint(repository_root: Path) -> str:
    paths = sorted((repository_root / "src" / "pipeguard").rglob("*.py"))
    paths.extend(sorted((repository_root / "src" / "pipeguard" / "specs").glob("*.json")))
    material = [
        (str(path.relative_to(repository_root)), _hash(path))
        for path in paths
        if path.is_file()
    ]
    return sha256_text(canonical_json(material))


def _capture_results(run_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_root.rglob("result.json")):
        result = _load(path)
        spec_id = result.get("spec_id")
        if spec_id is None:
            continue
        if spec_id in indexed:
            raise ValueError(f"multiple capture results found for specification {spec_id}")
        indexed[spec_id] = (path, result)
    return indexed


def _verify_capture_artifacts(result_path: Path, result: dict[str, Any]) -> None:
    case_directory = result_path.parent
    for artifact in result.get("artifacts", {}).values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("reference_path", "reference_sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            path_text = artifact.get(path_key)
            expected = artifact.get(hash_key)
            if path_text is None and expected is None:
                continue
            if not isinstance(path_text, str) or not isinstance(expected, str):
                raise ValueError(f"malformed artifact record in {result_path}")
            path = _relative_file(case_directory, path_text)
            if _hash(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")


def _selected_pgir(spec: dict[str, Any], capture: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    available = {
        key.split(":", 1)[1]: artifact
        for key, artifact in capture.get("artifacts", {}).items()
        if key.startswith("pgir:")
    }
    policy = spec["execution"]["symbol_policy"]
    declared = spec["execution"]["symbols"]
    if policy == "ONLY":
        if len(available) != 1:
            raise ValueError(f"{spec['spec_id']} expected exactly one PGIR symbol, got {sorted(available)}")
        names = sorted(available)
    elif policy == "EXACT":
        missing = set(declared) - set(available)
        if missing:
            raise ValueError(f"{spec['spec_id']} missing declared PGIR symbols {sorted(missing)}")
        names = sorted(declared)
    else:
        names = sorted(available)
    if not names:
        raise ValueError(f"{spec['spec_id']} selected no PGIR symbols")
    return [(name, available[name]) for name in names]


def _checker_metadata(repository_root: Path, engine: str, fingerprint: str, version: str) -> dict[str, Any]:
    return {
        "commit": git_commit(repository_root),
        "version": version,
        "engine": engine,
        "fingerprint": fingerprint,
    }


def _base_result(
    *,
    lock_sha256: str,
    catalog_sha256: str,
    spec: dict[str, Any],
    capture_path: Path,
    capture: dict[str, Any],
    run_root: Path,
    checker: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKER_RESULT_VERSION,
        "suite_lock_sha256": lock_sha256,
        "catalog_sha256": catalog_sha256,
        "kernel_id": spec["kernel_id"],
        "revision_id": spec["revision_id"],
        "spec_id": spec["spec_id"],
        "capture": {
            "result_path": str(capture_path.relative_to(run_root)),
            "result_sha256": _hash(capture_path),
            "fingerprint": capture["fingerprint"],
            "status": capture["status"],
        },
        "selection": {
            "selected_cta": spec["execution"]["selected_cta"],
            "claim_scope": spec["execution"]["claim_scope"],
            "symbol_policy": spec["execution"]["symbol_policy"],
        },
        "checker": checker,
        "status": status,
        "diagnostics": [],
        "timings": {},
    }


def _run_checker_worker(
    repository_root: Path,
    pgir_path: Path,
    engine: str,
    report_path: Path,
    timeout: float,
) -> tuple[str, str, float]:
    request = {
        "pgir_path": str(pgir_path),
        "engine": engine,
        "report_path": str(report_path),
    }
    request_path = report_path.with_suffix(".request.json")
    atomic_write(request_path, pretty_json(request))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path)]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return "CHECK_TIMEOUT", f"checker exceeded {timeout} seconds", time.monotonic() - started
    if process.returncode != 0:
        message = stderr[-4000:].strip() or stdout[-4000:].strip() or f"checker exited {process.returncode}"
        return "CHECK_ERROR", message, time.monotonic() - started
    return "CHECKED", "", time.monotonic() - started


def _worker(request_path: Path) -> int:
    request = _load(request_path)
    from pipeguard.checker import check_module
    from pipeguard.model import module_from_dict

    module = module_from_dict(_load(Path(request["pgir_path"])))
    report = check_module(module, engine=request["engine"])
    atomic_write(Path(request["report_path"]), pretty_json(report))
    return 0


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"code": code, "message": message}
    if details:
        diagnostic["details"] = details
    return diagnostic


def _checked_result(
    base: dict[str, Any],
    *,
    run_root: Path,
    capture_path: Path,
    symbol: str,
    artifact: dict[str, Any],
    report: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    pgir_path = _relative_file(capture_path.parent, artifact["path"])
    witness = report.get("witness")
    if witness is None and report.get("diagnostics"):
        witness = report["diagnostics"][0].get("witness")
    primary = report["diagnostics"][0]["code"] if report.get("diagnostics") else None
    result = dict(base)
    result["pgir"] = {
        "symbol": symbol,
        "path": str(pgir_path.relative_to(run_root)),
        "sha256": artifact["sha256"],
        "semantic_sha256": artifact["semantic_sha256"],
    }
    result["checker"] = dict(base["checker"], version=str(report.get("checker_version", "unknown")))
    result["actual"] = {
        "verdict": report["verdict"],
        "primary_diagnostic": primary,
        "witness_sha256": None if witness is None else sha256_text(canonical_json(witness)),
        "statistics": report.get("statistics", {}),
    }
    result["timings"] = {"check": elapsed}
    return result


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check captured Suite 3 PGIR in a process isolated from frontend lowering")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--corpus-root", type=Path, default=Path(__file__).parents[1] / "corpus" / "upstream")
    parser.add_argument("--suite-lock", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec-id", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.worker_request is None:
        if arguments.run_root is None or arguments.output is None:
            parser.error("--run-root and --output are required")
        if arguments.timeout <= 0:
            parser.error("--timeout must be positive")
        if len(arguments.spec_id) != len(set(arguments.spec_id)):
            parser.error("--spec-id values must be unique")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if arguments.worker_request is not None:
        return _worker(arguments.worker_request.resolve(strict=True))

    repository_root = arguments.repository_root.resolve(strict=True)
    corpus_root = arguments.corpus_root.resolve(strict=True)
    run_root = arguments.run_root.resolve(strict=True)
    output = arguments.output.resolve()
    lock_path = None if arguments.suite_lock is None else arguments.suite_lock.resolve(strict=True)
    catalog_path = None if arguments.catalog is None else arguments.catalog.resolve(strict=True)
    lock, catalog, _summary = validate_suite(corpus_root, lock_path=lock_path, catalog_path=catalog_path)
    lock_path = lock_path or corpus_root / "suite3" / "lock.json"
    catalog_path = catalog_path or corpus_root / "suite3" / "catalog.json"
    specs = {item["spec_id"]: item for item in catalog["specifications"]}
    selected_ids = set(arguments.spec_id) if arguments.spec_id else set(specs)
    missing = selected_ids - set(specs)
    if missing:
        raise ValueError(f"unknown suite specification IDs: {sorted(missing)}")

    captures = _capture_results(run_root)
    checker_fingerprint = _checker_fingerprint(repository_root)
    written: list[Path] = []
    for spec_id in sorted(selected_ids):
        spec = specs[spec_id]
        captured = captures.get(spec_id)
        if captured is None:
            raise ValueError(f"capture result missing for specification {spec_id}")
        capture_path, capture = captured
        _verify_capture_artifacts(capture_path, capture)
        checker = _checker_metadata(repository_root, spec["execution"]["engine"], checker_fingerprint, "not-run")
        base = _base_result(
            lock_sha256=_hash(lock_path),
            catalog_sha256=_hash(catalog_path),
            spec=spec,
            capture_path=capture_path,
            capture=capture,
            run_root=run_root,
            checker=checker,
            status="CAPTURE_NOT_AVAILABLE",
        )
        if capture["status"] != "CAPTURED":
            base["diagnostics"] = [_diagnostic(
                "UPSTREAM_CAPTURE_NOT_AVAILABLE",
                f"capture status is {capture['status']}; checker was not invoked",
                capture_diagnostic_codes=[
                    item.get("code") for item in capture.get("diagnostics", [])
                    if item.get("code") is not None
                ],
            )]
            result_path = output / safe_component(spec_id) / "capture-not-available.json"
            validate_checker_result(corpus_root, base)
            atomic_write(result_path, pretty_json(base))
            written.append(result_path)
            continue

        for symbol, artifact in _selected_pgir(spec, capture):
            pgir_path = _relative_file(capture_path.parent, artifact["path"])
            if _hash(pgir_path) != artifact["sha256"]:
                raise ValueError(f"PGIR hash mismatch for {spec_id}:{symbol}")
            result_directory = output / safe_component(spec_id)
            report_path = result_directory / f"{safe_component(symbol)}.checker-report.json"
            result_path = result_directory / f"{safe_component(symbol)}.result.json"
            if not arguments.force and result_path.is_file():
                previous = _load(result_path)
                if (
                    previous.get("capture", {}).get("result_sha256") == _hash(capture_path)
                    and previous.get("checker", {}).get("fingerprint") == checker_fingerprint
                    and previous.get("checker", {}).get("engine") == spec["execution"]["engine"]
                    and previous.get("pgir", {}).get("sha256") == artifact["sha256"]
                ):
                    validate_checker_result(corpus_root, previous)
                    written.append(result_path)
                    continue
            result_directory.mkdir(parents=True, exist_ok=True)
            status, message, elapsed = _run_checker_worker(
                repository_root,
                pgir_path,
                spec["execution"]["engine"],
                report_path,
                arguments.timeout,
            )
            if status == "CHECKED":
                report = _load(report_path)
                result = _checked_result(
                    dict(base, status="CHECKED"),
                    run_root=run_root,
                    capture_path=capture_path,
                    symbol=symbol,
                    artifact=artifact,
                    report=report,
                    elapsed=elapsed,
                )
            else:
                result = dict(base, status=status)
                result["pgir"] = {
                    "symbol": symbol,
                    "path": str(pgir_path.relative_to(run_root)),
                    "sha256": artifact["sha256"],
                    "semantic_sha256": artifact["semantic_sha256"],
                }
                result["diagnostics"] = [_diagnostic(
                    "UPSTREAM_CHECK_TIMEOUT" if status == "CHECK_TIMEOUT" else "UPSTREAM_CHECK_FAILED",
                    message,
                )]
                result["timings"] = {"check": elapsed}
            validate_checker_result(corpus_root, result)
            atomic_write(result_path, pretty_json(result))
            written.append(result_path)

    summary = {
        "schema_version": "pipeguard-upstream-check-summary-v1",
        "selected_specs": len(selected_ids),
        "result_files": len(written),
        "output": str(output),
    }
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
