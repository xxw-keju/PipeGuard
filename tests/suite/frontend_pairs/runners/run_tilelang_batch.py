from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TOOLS))

import tilelang_capture  # noqa: E402
from suite.frontend_pairs.catalog import PAIRS  # noqa: E402
from suite.frontend_pairs.sources.tilelang_cases import lower_tilelang_case  # noqa: E402
from tilelang import tvm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", action="append", type=int)
    args = parser.parse_args()

    output = Path(args.output)
    raw_root = output / "raw" / "tilelang"
    snapshot_root = output / "snapshots" / "tilelang"
    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.variant or range(12))
    records = []
    for case in PAIRS:
        if case.variant not in selected:
            continue
        raw_path = raw_root / f"{case.label}.tvm.json"
        snapshot_path = snapshot_root / f"{case.label}.json"
        record = {
            "label": case.label,
            "pair_id": case.pair_id,
            "rule_id": case.rule_id,
            "polarity": case.polarity,
            "family": case.family,
            "shape": case.shape,
        }
        try:
            raw_path.write_text(tvm.ir.save_json(
                lower_tilelang_case(case.family, case.variant, case.polarity)
            ))
            snapshot = tilelang_capture.capture_file(
                raw_path,
                source=f"suite2:{case.label}",
                command="suite2-native-tilelang-compile-only",
            )
            snapshot_path.write_text(json.dumps(snapshot, sort_keys=True))
            record.update({
                "status": "CAPTURED",
                "raw_path": str(raw_path.relative_to(output)),
                "snapshot_path": str(snapshot_path.relative_to(output)),
                "operations": len(snapshot["operations"]),
            })
        except Exception as error:
            diagnostic = error.diagnostic() if isinstance(error, tilelang_capture.CaptureError) else {
                "code": "TILELANG_NATIVE_LOWERING_FAILED",
                "message": str(error),
                "error_type": type(error).__name__,
            }
            record.update({
                "status": "UNSUPPORTED",
                "diagnostic": diagnostic,
                "raw_path": str(raw_path.relative_to(output)) if raw_path.exists() else None,
            })
        records.append(record)
        (output / "tilelang-results.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({"dsl": "tilelang", "counts": counts, "cases": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
