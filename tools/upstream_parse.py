#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from upstream_common import atomic_write, pretty_json  # noqa: E402


def _snapshot_to_pgir(dsl: str, snapshot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    checker_was_loaded = "pipeguard.checker" in sys.modules
    if dsl == "tilelang":
        from pipeguard.frontends.tilelang import tilelang_snapshot_to_module

        module = tilelang_snapshot_to_module(snapshot)
    else:
        from pipeguard.frontends.triton import triton_snapshot_to_module

        module = triton_snapshot_to_module(snapshot)
    from pipeguard.model import normalize_module, stable_digest

    normalized = normalize_module(module)
    if not checker_was_loaded and "pipeguard.checker" in sys.modules:
        raise RuntimeError("frontend-only parse imported pipeguard.checker")
    return normalized, stable_digest(normalized)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse one strict frontend snapshot into normalized PGIR")
    parser.add_argument("--dsl", choices=("tilelang", "triton"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--pgir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    snapshot = json.loads(arguments.snapshot.read_text())
    normalized, digest = _snapshot_to_pgir(arguments.dsl, snapshot)
    atomic_write(arguments.pgir, pretty_json(normalized))
    atomic_write(arguments.metadata, pretty_json({
        "checker_loaded": "pipeguard.checker" in sys.modules,
        "semantic_sha256": digest,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
