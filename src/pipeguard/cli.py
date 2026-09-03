from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .checker import check_module
from .frontends import FrontendError, load_tilelang_snapshot, load_triton_snapshot
from .model import PGIRError, load_module, normalize_module, stable_digest


def _dump(value: Any, pretty: bool = False) -> None:
    options = {"sort_keys": True, "ensure_ascii": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    print(json.dumps(value, **options))


def _empty_statistics(engine: str) -> dict[str, Any]:
    return {
        "actors": None, "events": None, "resources": None, "regions": None,
        "explored_steps": 0, "explored_states": 0, "engine": engine,
    }


def _internal(error: Exception, engine: str) -> dict[str, Any]:
    return {
        "verdict": "INTERNAL_ERROR", "checker_version": "0.2.0",
        "summary": "checker failed during validation or analysis",
        "diagnostics": [{"code": type(error).__name__, "message": str(error)}],
        "statistics": _empty_statistics(engine), "artifacts": {},
    }


def _load_failure(error: PGIRError | FrontendError, engine: str) -> dict[str, Any]:
    inconclusive = error.code in {"INCONCLUSIVE_PARITY", "UNRESOLVED_ASSUMPTIONS"}
    return {
        "verdict": "INCONCLUSIVE" if inconclusive else "UNSUPPORTED", "checker_version": "0.2.0",
        "summary": error.message, "diagnostics": [error.diagnostic()],
        "statistics": _empty_statistics(engine), "artifacts": {},
    }


def _load_input(path: str, frontend: str):
    if frontend == "pgir":
        return load_module(path)
    if frontend == "tilelang":
        return load_tilelang_snapshot(path)
    if frontend == "triton":
        return load_triton_snapshot(path)
    raise ValueError(f"unknown frontend: {frontend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeguard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "normalize", "check", "explain"):
        child = subparsers.add_parser(command)
        child.add_argument("input", help="path to a PGIR or frontend snapshot JSON document")
        child.add_argument(
            "--frontend",
            choices=("pgir", "tilelang", "triton"),
            default="pgir",
            help="input format (default: pgir)",
        )
        child.add_argument("--pretty", action="store_true", help="indent JSON output")
        if command in {"check", "explain"}:
            child.add_argument(
                "--engine", choices=("maximal", "exhaustive", "crosscheck"), default="maximal",
                help="execution engine (default: maximal)",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        module = _load_input(args.input, args.frontend)
        if args.command == "validate":
            result = {
                "valid": True, "schema_version": module.raw["schema_version"], "normalized_sha256": stable_digest(normalize_module(module)),
                "counts": {"actors": len(module.actors), "events": len(module.events), "regions": len(module.regions), "resources": len(module.resources)},
                "diagnostics": [],
            }
            _dump(result, args.pretty)
            return 0
        if args.command == "normalize":
            _dump(normalize_module(module), args.pretty)
            return 0
        result = check_module(module, engine=args.engine)
        if args.command == "explain":
            result = dict(result)
            witness = result.get("witness", {})
            result["explanation"] = {
                "verdict": result["summary"], "primary_reason": witness.get("category", "all supported checks discharged"),
                "repair": witness.get("repair"),
            }
        _dump(result, args.pretty)
        return 0 if result["verdict"] == "SAFE" else 1
    except (PGIRError, FrontendError) as error:
        if args.command == "validate":
            _dump({"valid": False, "schema_version": None, "diagnostics": [error.diagnostic()]}, args.pretty)
        else:
            _dump(_load_failure(error, getattr(args, "engine", "maximal")), args.pretty)
        return 2
    except Exception as error:  # Last-resort stable fail-closed boundary.
        _dump(_internal(error, getattr(args, "engine", "maximal")), args.pretty)
        return 3


if __name__ == "__main__":
    sys.exit(main())
