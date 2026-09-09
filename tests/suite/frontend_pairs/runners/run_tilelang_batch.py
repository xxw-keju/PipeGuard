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
    parser.add_argument("--family", action="append")
    parser.add_argument("--label", action="append")
    args = parser.parse_args()

    output = Path(args.output)
    raw_root = output / "raw" / "tilelang"
    snapshot_root = output / "snapshots" / "tilelang"
    result_path = output / "tilelang-results.json"
    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    selected_variants = set(args.variant) if args.variant else None
    selected_families = set(args.family) if args.family else None
    selected_labels = set(args.label) if args.label else None
    records_by_label = {
        record["label"]: record
        for record in (json.loads(result_path.read_text()) if result_path.exists() else [])
    }

    processed = 0
    for case in PAIRS:
        if selected_variants is not None and case.variant not in selected_variants:
            continue
        if selected_families is not None and case.family not in selected_families:
            continue
        if selected_labels is not None and case.label not in selected_labels:
            continue
        raw_path = raw_root / f"{case.label}.tvm.json"
        snapshot_path = snapshot_root / f"{case.label}.json"
        record = {
            "label": case.label,
            "pair_id": case.pair_id,
            "rule_id": case.rule_id,
            "polarity": case.polarity,
            "family": case.family,
            "scenario_id": case.scenario_id,
            "expected_stage": case.expected_stage,
            "shape": case.shape,
        }
        try:
            raw_path.write_text(tvm.ir.save_json(
                lower_tilelang_case(case.family, case.scenario_id, case.variant, case.polarity)
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
        records_by_label[case.label] = record
        processed += 1
        result_path.write_text(json.dumps(
            sorted(records_by_label.values(), key=lambda item: item["label"]),
            indent=2,
            sort_keys=True,
        ))

    counts = {}
    for record in records_by_label.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({
        "dsl": "tilelang",
        "processed": processed,
        "recorded": len(records_by_label),
        "counts": counts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
