from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .checker import check_module
from .frontends import (
    FrontendError,
    load_tilelang_snapshot,
    load_triton_snapshot,
    semantic_digest,
)
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
    compare = subparsers.add_parser("compare")
    compare.add_argument("left", help="first PGIR or frontend snapshot")
    compare.add_argument("right", help="second PGIR or frontend snapshot")
    compare.add_argument("--left-frontend", choices=("pgir", "tilelang", "triton"), required=True)
    compare.add_argument("--right-frontend", choices=("pgir", "tilelang", "triton"), required=True)
    compare.add_argument(
        "--engine", choices=("maximal", "exhaustive", "crosscheck"), default="crosscheck",
        help="checker engine (default: crosscheck)",
    )
    compare.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def _comparison_result(module, engine: str) -> dict[str, Any]:
    result = check_module(module, engine=engine)
    diagnostic = result.get("diagnostics", [{}])
    return {
        "semantic_sha256": semantic_digest(module),
        "verdict": result["verdict"],
        "primary_diagnostic": diagnostic[0].get("code") if diagnostic else None,
        "n_ext": result["statistics"].get("n_ext"),
        "t_eff": result["statistics"].get("t_eff"),
        "scheduler_work_units": result["statistics"].get("scheduler_work_units"),
        "pdg_edges": result["statistics"].get("pdg_edges"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare":
            left = _load_input(args.left, args.left_frontend)
            right = _load_input(args.right, args.right_frontend)
            left_result = _comparison_result(left, args.engine)
            right_result = _comparison_result(right, args.engine)
            semantic_equivalent = left_result["semantic_sha256"] == right_result["semantic_sha256"]
            checker_equivalent = (
                left_result["verdict"],
                left_result["primary_diagnostic"],
            ) == (
                right_result["verdict"],
                right_result["primary_diagnostic"],
            )
            result = {
                "equivalent": semantic_equivalent and checker_equivalent,
                "semantic_equivalent": semantic_equivalent,
                "checker_equivalent": checker_equivalent,
                "left": left_result,
                "right": right_result,
            }
            _dump(result, args.pretty)
            return 0 if result["equivalent"] else 1
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
