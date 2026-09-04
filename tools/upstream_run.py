#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from upstream_common import (
    RUN_RESULT_VERSION,
    TOOL_VERSION,
    atomic_write,
    canonical_json,
    git_commit,
    pretty_json,
    require_locked_checkout,
    sha256_bytes,
    sha256_text,
)


DEFAULT_TILELANG_PYTHON = Path("/usr/local/miniconda3/envs/tl-pg/bin/python")
DEFAULT_TRITON_PYTHON = Path("/usr/local/miniconda3/envs/triton-pg/bin/python")
DEFAULT_TRITON_PTXAS = Path("/usr/local/cuda-12.9/bin/ptxas")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _artifacts_match(case_directory: Path, result: dict[str, Any]) -> bool:
    for artifact in result.get("artifacts", {}).values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("reference_path", "reference_sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            relative_text = artifact.get(path_key)
            expected = artifact.get(hash_key)
            if relative_text is None and expected is None:
                continue
            if not isinstance(relative_text, str) or not isinstance(expected, str):
                return False
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = case_directory / relative
            if not path.is_file() or _file_hash(path) != expected:
                return False
    return True


def _worker_environment(dsl: str, arguments: argparse.Namespace, case_directory: Path) -> dict[str, str]:
    environment = {
        "HOME": str(case_directory / "home"),
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(arguments.hash_seed),
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    if dsl == "triton":
        environment["TRITON_CACHE_DIR"] = str(case_directory / "triton-cache")
        environment["TRITON_PTXAS_PATH"] = str(arguments.triton_ptxas)
        environment["TRITON_KERNEL_OVERRIDE"] = "0"
        environment["TRITON_STORE_BINARY_ONLY"] = "0"
    return environment


def _fingerprint(
    dsl: str,
    manifest: dict[str, Any],
    recipe: dict[str, Any],
    source_root: Path,
    interpreter: Path,
    worker: Path,
    parser: Path,
    capture: Path,
    extra_tool: dict[str, Any],
) -> str:
    material = {
        "tool_version": TOOL_VERSION,
        "dsl": dsl,
        "source_commit": git_commit(source_root),
        "source_sha256": manifest["upstream"]["source_sha256"],
        "capsule_sha256": manifest["capsule"]["sha256"],
        "manifest": manifest,
        "recipe": recipe,
        "interpreter": str(interpreter.resolve(strict=True)),
        "interpreter_sha256": _file_hash(interpreter.resolve(strict=True)),
        "runner_sha256": _file_hash(Path(__file__).resolve(strict=True)),
        "worker_sha256": _file_hash(worker),
        "parser_sha256": _file_hash(parser),
        "capture_sha256": _file_hash(capture),
        "extra_tool": extra_tool,
    }
    return sha256_text(canonical_json(material))


def _selected_work(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    selected_dsls = ("tilelang", "triton") if arguments.dsl == "all" else (arguments.dsl,)
    work: list[dict[str, Any]] = []
    for dsl in selected_dsls:
        corpus_name = "tilelang-2c31e071" if dsl == "tilelang" else "triton-61c02bcd"
        index = _load(arguments.corpus_root / corpus_name / "source-index.json")
        for candidate in index["candidates"]:
            if arguments.candidate and candidate["candidate_id"] not in arguments.candidate:
                continue
            manifest = _load(arguments.corpus_root / candidate["manifest_path"])
            for recipe_path_text in candidate["recipe_paths"]:
                recipe_path = arguments.corpus_root / recipe_path_text
                recipe = _load(recipe_path)
                if not recipe.get("lowerable") and not arguments.include_unresolved:
                    continue
                work.append({
                    "dsl": dsl,
                    "candidate": candidate,
                    "manifest": manifest,
                    "recipe": recipe,
                    "recipe_path": recipe_path,
                })
    return sorted(work, key=lambda item: (
        item["dsl"], item["candidate"]["candidate_id"], item["recipe"]["recipe_id"]
    ))


class PgirParseTimeout(RuntimeError):
    pass


def _parse_snapshots(
    dsl: str,
    case_directory: Path,
    result: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    for key, artifact in list(result["artifacts"].items()):
        if not key.startswith("snapshot:"):
            continue
        symbol = key.split(":", 1)[1]
        snapshot_path = case_directory / artifact["path"]
        pgir_path = case_directory / f"pgir-{symbol}.json"
        metadata_path = case_directory / f"pgir-{symbol}.metadata.json"
        command = [
            sys.executable,
            str(arguments.parser),
            "--dsl", dsl,
            "--snapshot", str(snapshot_path),
            "--pgir", str(pgir_path),
            "--metadata", str(metadata_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=arguments.repository_root,
            env={
                "HOME": str(case_directory / "home"),
                "LC_ALL": "C.UTF-8",
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": str(arguments.hash_seed),
                "PYTHONNOUSERSITE": "1",
                "TZ": "UTC",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=arguments.parse_timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise PgirParseTimeout(
                f"PGIR parse for {symbol} exceeded {arguments.parse_timeout} seconds"
            ) from error
        atomic_write(case_directory / f"parser-{symbol}.stdout", stdout[-1_000_000:])
        atomic_write(case_directory / f"parser-{symbol}.stderr", stderr[-1_000_000:])
        if process.returncode != 0:
            raise RuntimeError(
                f"PGIR parser exited {process.returncode}: {stderr[-4000:].strip()}"
            )
        metadata = _load(metadata_path)
        if metadata.get("checker_loaded") is not False:
            raise RuntimeError("frontend-only parser loaded pipeguard.checker")
        result["artifacts"][f"pgir:{symbol}"] = {
            "path": pgir_path.name,
            "sha256": _file_hash(pgir_path),
            "metadata_path": metadata_path.name,
            "metadata_sha256": _file_hash(metadata_path),
            "semantic_sha256": metadata["semantic_sha256"],
        }


def _heartbeat_payload(
    started: float,
    work: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    current: dict[str, Any] | None,
    resumed: int,
) -> dict[str, Any]:
    status_by_dsl: dict[str, Counter[str]] = {"tilelang": Counter(), "triton": Counter()}
    for result in completed:
        status_by_dsl[result["dsl"]][result["status"]] += 1
    elapsed = max(0.0, time.monotonic() - started)
    done = len(completed)
    throughput = done / elapsed if elapsed else 0.0
    remaining = len(work) - done
    selected_by_dsl = Counter(item["dsl"] for item in work)
    completed_by_dsl = Counter(result["dsl"] for result in completed)
    return {
        "event": "progress",
        "elapsed_seconds": round(elapsed, 3),
        "selected_total": len(work),
        "completed": done,
        "remaining": remaining,
        "resumed": resumed,
        "selected_by_dsl": dict(sorted(selected_by_dsl.items())),
        "completed_by_dsl": dict(sorted(completed_by_dsl.items())),
        "remaining_by_dsl": {
            dsl: selected_by_dsl[dsl] - completed_by_dsl[dsl]
            for dsl in sorted(selected_by_dsl)
        },
        "throughput_per_second": round(throughput, 6),
        "eta_seconds": round(remaining / throughput, 3) if throughput else None,
        "current": None if current is None else {
            "dsl": current["dsl"],
            "candidate_id": current["candidate"]["candidate_id"],
            "recipe_id": current["recipe"]["recipe_id"],
            "phase": "worker",
        },
        "tilelang": dict(sorted(status_by_dsl["tilelang"].items())),
        "triton": dict(sorted(status_by_dsl["triton"].items())),
    }


def _emit_progress(payload: dict[str, Any], progress_path: Path) -> None:
    line = canonical_json(payload)
    print(line, file=sys.stderr, flush=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timeout_result(request: dict[str, Any], timeout: float) -> dict[str, Any]:
    return {
        "schema_version": RUN_RESULT_VERSION,
        "fingerprint": request["fingerprint"],
        "dsl": request["dsl"],
        "candidate_id": request["manifest"]["candidate_id"],
        "recipe_id": request["recipe"]["recipe_id"],
        "status": "TIMEOUT",
        "phase": "worker",
        "diagnostics": [{
            "code": "UPSTREAM_WORKER_TIMEOUT",
            "message": f"worker exceeded {timeout} seconds",
        }],
        "artifacts": {},
        "timings": {"timeout": timeout},
    }


def _run_one(
    item: dict[str, Any],
    arguments: argparse.Namespace,
    run_root: Path,
    heartbeat,
) -> tuple[dict[str, Any], bool]:
    dsl = item["dsl"]
    interpreter = arguments.tilelang_python if dsl == "tilelang" else arguments.triton_python
    source_root = arguments.tilelang_root if dsl == "tilelang" else arguments.triton_root
    capture = arguments.repository_root / "tools" / f"{dsl}_capture.py"
    extra_tool: dict[str, Any] = {
        "worker_timeout": arguments.timeout,
        "parse_timeout": arguments.parse_timeout,
    }
    if dsl == "triton":
        version = subprocess.run(
            [str(arguments.triton_ptxas), "--version"], capture_output=True, text=True, check=True
        ).stdout
        extra_tool.update({"ptxas_path": str(arguments.triton_ptxas), "ptxas_version": version})
    fingerprint = _fingerprint(
        dsl, item["manifest"], item["recipe"], source_root,
        interpreter, arguments.worker, arguments.parser, capture, extra_tool,
    )
    case_directory = run_root / dsl / item["candidate"]["candidate_id"] / item["recipe"]["recipe_id"]
    result_path = case_directory / "result.json"
    if not arguments.force and result_path.exists():
        previous = _load(result_path)
        if (
            previous.get("fingerprint") == fingerprint
            and previous.get("status") in {
                "CAPTURED", "UNSUPPORTED", "PROFILE_SKIPPED", "IMPORT_ERROR",
                "LOWERING_ERROR", "TIMEOUT", "INFRA_ERROR",
            }
            and _artifacts_match(case_directory, previous)
        ):
            return previous, True
    case_directory.mkdir(parents=True, exist_ok=True)
    request = {
        "fingerprint": fingerprint,
        "dsl": dsl,
        "corpus_root": str(arguments.corpus_root),
        "source_root": str(source_root),
        "manifest": item["manifest"],
        "recipe": item["recipe"],
    }
    request_path = case_directory / "request.json"
    atomic_write(request_path, pretty_json(request))
    command = [str(interpreter), str(arguments.worker), "--request", str(request_path), "--output", str(case_directory)]
    process = subprocess.Popen(
        command,
        cwd=arguments.repository_root,
        env=_worker_environment(dsl, arguments, case_directory),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + arguments.timeout
    while process.poll() is None:
        if time.monotonic() >= deadline:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result = _timeout_result(request, arguments.timeout)
            atomic_write(result_path, pretty_json(result))
            return result, False
        heartbeat()
        time.sleep(min(0.25, max(0.01, arguments.progress_interval / 10)))
    stdout, stderr = process.communicate()
    atomic_write(case_directory / "worker.stdout", stdout[-1_000_000:])
    atomic_write(case_directory / "worker.stderr", stderr[-1_000_000:])
    if not result_path.exists():
        result = {
            **_timeout_result(request, arguments.timeout),
            "status": "INFRA_ERROR",
            "diagnostics": [{
                "code": "UPSTREAM_WORKER_RESULT_MISSING",
                "message": f"worker exited {process.returncode} without result.json",
            }],
            "timings": {},
        }
    else:
        result = _load(result_path)
    if result.get("status") == "CAPTURED":
        try:
            _parse_snapshots(dsl, case_directory, result, arguments)
            result["phase"] = "pgir_parse"
        except PgirParseTimeout as error:
            result["status"] = "TIMEOUT"
            result["phase"] = "pgir_parse"
            result["diagnostics"].append({
                "code": "UPSTREAM_PGIR_PARSE_TIMEOUT",
                "message": str(error),
                "details": {"timeout": arguments.parse_timeout},
            })
        except Exception as error:
            result["status"] = "UNSUPPORTED"
            result["phase"] = "pgir_parse"
            result["diagnostics"].append({
                "code": "UPSTREAM_PGIR_PARSE_FAILED",
                "message": str(error),
                "details": {"error_type": type(error).__name__},
            })
    result["checker_loaded"] = "pipeguard.checker" in sys.modules
    atomic_write(result_path, pretty_json(result))
    return result, False


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked upstream kernels through native lowering and frontend-only PGIR parsing")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--corpus-root", type=Path, default=Path(__file__).parents[1] / "corpus" / "upstream")
    parser.add_argument("--tilelang-root", type=Path, required=True)
    parser.add_argument("--triton-root", type=Path, required=True)
    parser.add_argument("--tilelang-python", type=Path, default=DEFAULT_TILELANG_PYTHON)
    parser.add_argument("--triton-python", type=Path, default=DEFAULT_TRITON_PYTHON)
    parser.add_argument("--triton-ptxas", type=Path, default=DEFAULT_TRITON_PTXAS)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("upstream_worker.py"))
    parser.add_argument("--parser", type=Path, default=Path(__file__).with_name("upstream_parse.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsl", choices=("tilelang", "triton", "all"), default="all")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--include-unresolved", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--parse-timeout", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=150.0)
    parser.add_argument("--hash-seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    if arguments.timeout <= 0 or arguments.parse_timeout <= 0 or arguments.progress_interval <= 0:
        parser.error("timeouts and progress interval must be positive")
    if arguments.jobs < 1:
        parser.error("jobs must be positive")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    arguments.repository_root = arguments.repository_root.resolve(strict=True)
    arguments.corpus_root = arguments.corpus_root.resolve(strict=True)
    arguments.worker = arguments.worker.resolve(strict=True)
    arguments.parser = arguments.parser.resolve(strict=True)
    arguments.tilelang_root = arguments.tilelang_root.resolve(strict=True)
    arguments.triton_root = arguments.triton_root.resolve(strict=True)
    if arguments.dsl in {"tilelang", "all"}:
        require_locked_checkout("tilelang", arguments.tilelang_root)
    if arguments.dsl in {"triton", "all"}:
        require_locked_checkout("triton", arguments.triton_root)
    arguments.output.mkdir(parents=True, exist_ok=True)
    work = _selected_work(arguments)
    run_material = {
        "tool_version": TOOL_VERSION,
        "selected": [(item["dsl"], item["candidate"]["candidate_id"], item["recipe"]["recipe_id"]) for item in work],
        "hash_seed": arguments.hash_seed,
    }
    run_id = sha256_text(canonical_json(run_material))[:16]
    run_root = arguments.output / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "progress.jsonl"
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    resumed = 0
    last_report = [float("-inf")]
    active: dict[int, dict[str, Any]] = {}
    state_lock = threading.Lock()

    def heartbeat(force: bool = False):
        nonlocal resumed
        with state_lock:
            now = time.monotonic()
            if not force and now - last_report[0] < arguments.progress_interval:
                return
            current = active[min(active)] if active else None
            payload = _heartbeat_payload(started, work, list(completed), current, resumed)
            payload["active_workers"] = len(active)
            _emit_progress(payload, progress_path)
            last_report[0] = now

    def execute(index: int, item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        with state_lock:
            active[index] = item
        try:
            return _run_one(item, arguments, run_root, heartbeat)
        finally:
            with state_lock:
                active.pop(index, None)

    heartbeat(True)
    worker_count = min(arguments.jobs, max(1, len(work)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(execute, index, item): index
            for index, item in enumerate(work)
        }
        for future in concurrent.futures.as_completed(futures):
            result, was_resumed = future.result()
            with state_lock:
                completed.append(result)
                resumed += int(was_resumed)
            heartbeat()
    heartbeat(True)
    summary = {
        "schema_version": "pipeguard-upstream-summary-v1",
        "run_id": run_id,
        "checker_loaded": "pipeguard.checker" in sys.modules,
        "selected_total": len(work),
        "completed": len(completed),
        "resumed": resumed,
        "tilelang": dict(sorted(Counter(r["status"] for r in completed if r["dsl"] == "tilelang").items())),
        "triton": dict(sorted(Counter(r["status"] for r in completed if r["dsl"] == "triton").items())),
    }
    atomic_write(run_root / "summary.json", pretty_json(summary))
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
