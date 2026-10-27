"""Suite 2 admission gate 1: structural fingerprint uniqueness audit.

Reads a suite2 full-recapture output directory (layout:
``<run-dir>/cases/<case_id>/snapshot.json`` + ``normalized-spec.json``),
fingerprints every case's captured protocol structure, groups cases into
cells keyed by (dsl, primitive, result_class), and reports per-cell
fingerprint uniqueness.

Gate semantics
--------------
Default (strict): a cell PASSes only if every variant in the cell has a
distinct structural fingerprint (``unique_fingerprints == n``).

``--min-unique N`` relaxes the gate to ``unique_fingerprints >= N``.
``--min-unique 2`` detects only fully-degenerate cells (all variants share
one skeleton), which is the headline finding the redesign is addressing.

``--baseline old-report.json`` additionally diffs this audit against a
previous report (cell status / uniqueness changes).

Exit status: 0 if every cell PASSes, 1 otherwise, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pipeguard-suite2-structural-audit-v1"
OPERATION_TYPE_KEYS = ("op", "operation", "type", "kind")


def operation_type(operation: dict[str, Any]) -> str:
    """Return the operation's type label (first non-empty of op/operation/type/kind)."""
    for key in OPERATION_TYPE_KEYS:
        value = operation.get(key)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"


def fingerprint_material(snapshot: dict[str, Any]) -> str:
    """The exact structural-fingerprint recipe (documented; keep in sync with ADMISSION-GATES.md).

    Material = three '|'-joined sections over the capture snapshot:
      1. ``ops=`` comma-joined sequence of operation types from
         ``snapshot["operations"]`` (order preserved; type = first non-empty
         of each op's ``op``/``operation``/``type``/``kind`` field).
      2. ``actors=`` the number of entries in ``snapshot["actors"]``.
      3. ``resource_kinds=`` comma-joined sorted multiset of the ``kind``
         field of ``snapshot["resources"]`` entries.

    Deliberately EXCLUDED (shape-level, not structure-level): region offsets
    and extents, transfer byte counts, per-op instance lists, barrier arrival
    counts, and all evidence/id fields. Two variants that differ only in
    those collapse to one fingerprint by design.
    """
    operations = snapshot.get("operations") or []
    ops = ",".join(operation_type(op) for op in operations if isinstance(op, dict))
    actors = snapshot.get("actors") or []
    resources = snapshot.get("resources") or []
    kinds = sorted(
        str(resource.get("kind", "")) for resource in resources if isinstance(resource, dict)
    )
    return f"ops={ops}|actors={len(actors)}|resource_kinds={','.join(kinds)}"


def structural_fingerprint(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(fingerprint_material(snapshot).encode("utf-8")).hexdigest()


def load_cases(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cases_dir = run_dir / "cases"
    if not cases_dir.is_dir():
        raise RuntimeError(f"not a suite2 run directory (missing {cases_dir})")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        spec_path = case_dir / "normalized-spec.json"
        snapshot_path = case_dir / "snapshot.json"
        if not spec_path.is_file():
            skipped.append({"case_id": case_dir.name, "reason": "MISSING_NORMALIZED_SPEC"})
            continue
        if not snapshot_path.is_file():
            skipped.append({"case_id": case_dir.name, "reason": "MISSING_SNAPSHOT"})
            continue
        spec = json.loads(spec_path.read_text())
        snapshot = json.loads(snapshot_path.read_text())
        rows.append({
            "case_id": spec.get("case_id", case_dir.name),
            "dsl": str(spec.get("dsl", "UNKNOWN")),
            "primitive": str(spec.get("primitive", "UNKNOWN")),
            "result_class": str(spec.get("result_class", "UNKNOWN")),
            "fingerprint": structural_fingerprint(snapshot),
        })
    return rows, skipped


def audit(rows: list[dict[str, Any]], min_unique: int | None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[":".join((row["dsl"], row["primitive"], row["result_class"]))].append(row)
    cells: dict[str, dict[str, Any]] = {}
    for key, members in sorted(grouped.items()):
        fingerprints: dict[str, list[str]] = defaultdict(list)
        for member in members:
            fingerprints[member["fingerprint"]].append(member["case_id"])
        unique = len(fingerprints)
        n = len(members)
        required = n if min_unique is None else min(min_unique, n)
        cells[key] = {
            "n": n,
            "unique_fingerprints": unique,
            "required_unique": required,
            "status": "PASS" if unique >= required else "FAIL",
            "fully_degenerate": unique == 1 and n > 1,
            "fingerprints": {
                fingerprint: sorted(case_ids)
                for fingerprint, case_ids in sorted(fingerprints.items())
            },
        }
    return cells


def diff_baseline(baseline: dict[str, Any], cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    old_cells: dict[str, Any] = baseline.get("cells", {})
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(old_cells) & set(cells)):
        old, new = old_cells[key], cells[key]
        fields = ("status", "unique_fingerprints", "n")
        if any(old.get(field) != new.get(field) for field in fields):
            changed[key] = {
                "old": {field: old.get(field) for field in fields},
                "new": {field: new.get(field) for field in fields},
            }
    return {
        "baseline_schema_version": baseline.get("schema_version"),
        "added_cells": sorted(set(cells) - set(old_cells)),
        "removed_cells": sorted(set(old_cells) - set(cells)),
        "changed_cells": changed,
    }


def render_table(cells: dict[str, dict[str, Any]]) -> str:
    header = f"{'cell (dsl:primitive:result_class)':<42} {'n':>3} {'unique':>6} {'req':>4} {'degenerate':>10} {'status':>6}"
    lines = [header, "-" * len(header)]
    for key, cell in sorted(cells.items()):
        lines.append(
            f"{key:<42} {cell['n']:>3} {cell['unique_fingerprints']:>6} "
            f"{cell['required_unique']:>4} {'yes' if cell['fully_degenerate'] else 'no':>10} {cell['status']:>6}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="suite2 full-recapture output directory")
    parser.add_argument("--min-unique", type=int, default=None,
                        help="PASS requires unique_fingerprints >= N (default: == n, i.e. strict)")
    parser.add_argument("--baseline", type=Path, default=None, help="previous audit report JSON to diff against")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here (default: no file)")
    args = parser.parse_args()
    if args.min_unique is not None and args.min_unique < 1:
        parser.error("--min-unique must be >= 1")

    run_dir = args.run_dir.resolve()
    rows, skipped = load_cases(run_dir)
    cells = audit(rows, args.min_unique)
    failing = sorted(key for key, cell in cells.items() if cell["status"] == "FAIL")
    degenerate = sorted(key for key, cell in cells.items() if cell["fully_degenerate"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "gate": "unique_fingerprints == n" if args.min_unique is None else f"unique_fingerprints >= {args.min_unique}",
        "recipe": {
            "hash": "sha256 over fingerprint_material (see tools/suite2_structural_audit.py)",
            "material": "ops=<comma-joined op types>|actors=<actor count>|resource_kinds=<sorted resource kinds>",
            "operation_type_keys": list(OPERATION_TYPE_KEYS),
            "excluded": "region offsets/extents, byte counts, instance lists, arrival counts, evidence/id fields",
        },
        "cases": len(rows),
        "skipped_cases": skipped,
        "cells": cells,
        "failing_cells": failing,
        "fully_degenerate_cells": degenerate,
        "summary": {
            "cells": len(cells),
            "pass": sum(cell["status"] == "PASS" for cell in cells.values()),
            "fail": len(failing),
            "fully_degenerate": len(degenerate),
        },
    }
    if args.baseline is not None:
        report["baseline_diff"] = diff_baseline(json.loads(args.baseline.read_text()), cells)

    print(render_table(cells))
    print()
    print(json.dumps(report["summary"], sort_keys=True))
    if skipped:
        print(f"WARNING: skipped {len(skipped)} case dirs (missing snapshot or spec)")
    if failing:
        print("FAILING CELLS:")
        for key in failing:
            print(f"  {key}")
    if args.baseline is not None:
        print("BASELINE DIFF:")
        print(json.dumps(report["baseline_diff"], indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {args.out}")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
