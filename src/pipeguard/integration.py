"""Optional TileLang integration, isolated from the caller's compiler process.

This exposes the existing 0.2 checker, not the in-progress semantic refactor.
No device kernel is launched. The deadline covers capture and checking.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any


def _report(verdict: str, code: str, message: str) -> dict[str, Any]:
    return {"verdict": verdict, "checker_version": "0.2.0",
            "summary": message, "diagnostics": [{"code": code, "message": message}],
            "statistics": {}, "artifacts": {}}


def _scope(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    result["scope"] = {
        "kind": "selected-cta", "launch_grid": request["launch_grid"],
        "selected_cta": request["selected_cta"], "source": request["source"],
        "source_ir_sha256": hashlib.sha256(request["ir_json"].encode()).hexdigest(),
        "cross_cta_checked": False, "device_code_checked": False,
        "checker_profile": "experimental-0.2",
    }
    return result


def check_tilelang_ir(
    ir_json: str, *, launch_grid: tuple[int, ...] | list[int],
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    source: str = "tilelang", descriptors: dict[str, Any] | None = None,
    scalar_values: dict[str, int] | None = None,
    pointer_table_allocations: dict[str, list[str]] | None = None,
    buffer_values: dict[str, list[int]] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Capture one lowered device IRModule and return a checker report.

    Inputs are exact recorded launch values, not sample/default bindings.
    ``descriptors`` maps host descriptor names to shape/stride evidence.
    All evidence is bound to the serialized IR hash in the worker. An absent
    binding that capture needs produces UNSUPPORTED. A timeout produces
    INCONCLUSIVE. No device-code correctness or numerical claim is made.
    """
    if not isinstance(ir_json, str):
        raise TypeError("ir_json must be tvm.ir.save_json(device_mod)")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    grid = list(launch_grid)
    cta = list(selected_cta)
    if not 1 <= len(grid) <= 3 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("launch_grid must contain one to three positive integers")
    grid += [1] * (3 - len(grid))
    if len(cta) != 3 or any(type(n) is not int or not 0 <= n < grid[i] for i, n in enumerate(cta)):
        raise ValueError("selected_cta must be three indices inside launch_grid")
    request = dict(ir_json=ir_json, launch_grid=grid, selected_cta=cta,
                   source=source, descriptors=descriptors or {},
                   scalar_values=scalar_values or {},
                   pointer_table_allocations=pointer_table_allocations or {},
                   buffer_values=buffer_values or {})
    with tempfile.TemporaryDirectory(prefix="pipeguard-") as tmp:
        directory = Path(tmp)
        request_path, result_path = directory / "request.json", directory / "result.json"
        request_path.write_text(json.dumps(request))
        # Also works from an editable/source installation without importing
        # a compiler into the checker library at module-import time.
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        with (directory / "worker.log").open("w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "pipeguard.integration", str(request_path), str(result_path)],
                stdout=log, stderr=log, env=env, start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return _scope(_report("INCONCLUSIVE", "CAPTURE_CHECK_TIMEOUT",
                                      f"Capture and checking exceeded {timeout:g} seconds"), request)
            if process.returncode != 0 or not result_path.exists():
                log.seek(0)
                result = _report("INTERNAL_ERROR", "CHECKER_WORKER_FAILED", log.read()[-4000:])
            else:
                result = json.loads(result_path.read_text())
        return _scope(result, request)


def _worker(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from .capture.tilelang_capture import CaptureError, capture_file
    from .checker import check_module
    from .frontends import FrontendError, tilelang_snapshot_to_module
    from .model import PGIRError

    ir = directory / "device.json"
    ir.write_text(request["ir_json"])
    digest = hashlib.sha256(request["ir_json"].encode()).hexdigest()
    descriptors = directory / "descriptors.json"
    descriptors.write_text(json.dumps({
        "evidence_version": "tilelang-tma-descriptor-evidence-v1",
        "source_ir_sha256": digest, "descriptors": request["descriptors"],
    }))
    bindings = directory / "bindings.json"
    bindings.write_text(json.dumps({
        "evidence_version": "tilelang-binding-evidence-v1", "source_ir_sha256": digest,
        "scalar_values": request["scalar_values"],
        "pointer_table_allocations": request["pointer_table_allocations"],
        "buffer_values": request["buffer_values"],
        "provenance": {"source": request["source"], "policy": "exact caller bindings"},
    }))
    try:
        snapshot = capture_file(ir, request["source"], "pipeguard.integration",
                                descriptors, bindings, launch_grid=request["launch_grid"],
                                selected_cta=request["selected_cta"])
        return check_module(tilelang_snapshot_to_module(snapshot))
    except (CaptureError, FrontendError, PGIRError) as error:
        verdict = "INCONCLUSIVE" if error.code in {"INCONCLUSIVE_PARITY", "UNRESOLVED_ASSUMPTIONS"} else "UNSUPPORTED"
        result = _report(verdict, error.code, error.message)
        result["diagnostics"] = [error.diagnostic()]
        return result


if __name__ == "__main__":
    request_path, result_path = map(Path, sys.argv[1:])
    try:
        result = _worker(json.loads(request_path.read_text()), request_path.parent)
    except Exception as error:
        result = _report("INTERNAL_ERROR", type(error).__name__, str(error))
    result_path.write_text(json.dumps(result))
