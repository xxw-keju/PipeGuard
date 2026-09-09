from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(ROOT / "src"))

from pipeguard.checker import check_module  # noqa: E402
from pipeguard.frontends.common import FrontendError  # noqa: E402
from pipeguard.model import PGIRError  # noqa: E402
from pipeguard.frontends.tilelang import tilelang_snapshot_to_module  # noqa: E402
from pipeguard.frontends.triton import triton_snapshot_to_module  # noqa: E402
from suite.frontend_pairs.catalog import PAIRS  # noqa: E402


CONVERTERS = {
    "tilelang": tilelang_snapshot_to_module,
    "triton": triton_snapshot_to_module,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    root = Path(args.artifacts)
    expected = {case.label: case for case in PAIRS}
    summaries: dict[str, list[dict]] = {}

    for dsl, converter in CONVERTERS.items():
        pgir_root = root / "pgir" / dsl
        pgir_root.mkdir(parents=True, exist_ok=True)
        capture_records = {
            record["label"]: record
            for record in json.loads((root / f"{dsl}-results.json").read_text())
        }
        records = []
        for label, case in expected.items():
            capture = capture_records[label]
            source_limitation = getattr(case, f"{dsl}_source_limitation")
            record = {
                "label": label,
                "pair_id": case.pair_id,
                "rule_id": case.rule_id,
                "polarity": case.polarity,
                "expected_stage": case.expected_stage,
                "expected_verdict": case.expected_verdict,
                "expected_diagnostic": case.expected_diagnostic,
                "capture_status": capture["status"],
            }
            if source_limitation is not None:
                record["source_limitation"] = source_limitation
            if capture["status"] != "CAPTURED":
                record["status"] = (
                    "SOURCE_UNREPRESENTABLE"
                    if source_limitation is not None
                    else "UNSUPPORTED"
                )
                record["matches_expected"] = False
                record["diagnostic"] = capture["diagnostic"]
            else:
                snapshot_path = root / capture["snapshot_path"]
                try:
                    module = converter(json.loads(snapshot_path.read_text()))
                    pgir_path = pgir_root / f"{label}.json"
                    pgir_path.write_text(json.dumps(module.raw, indent=2, sort_keys=True))
                    report = check_module(module, engine="maximal")
                    actual_diagnostic = (
                        report.get("diagnostics", [{}])[0].get("code")
                        if report.get("diagnostics") else None
                    )
                    record.update({
                        "status": (
                            "SOURCE_UNREPRESENTABLE"
                            if source_limitation is not None
                            else "CHECKED"
                        ),
                        "actual_verdict": report["verdict"],
                        "actual_diagnostic": actual_diagnostic,
                        "pgir_path": str(pgir_path.relative_to(root)),
                        "matches_expected": (
                            source_limitation is None
                            and case.expected_stage == "CHECKER"
                            and report["verdict"] == case.expected_verdict
                            and (
                                case.expected_diagnostic is None
                                or actual_diagnostic == case.expected_diagnostic
                            )
                        ),
                        "events": report["statistics"]["events"],
                        "n_ext": report["statistics"]["n_ext"],
                        "t_eff": report["statistics"]["t_eff"],
                    })
                except (PGIRError, FrontendError) as error:
                    code = error.code
                    model_rejected = isinstance(error, PGIRError)
                    expected_rejection = (
                        case.expected_stage == "MODEL"
                        and code == case.expected_diagnostic
                    )
                    record.update({
                        "status": (
                            "MODEL_REJECTED"
                            if expected_rejection
                            else "SOURCE_UNREPRESENTABLE"
                            if source_limitation is not None
                            else "MODEL_REJECTED"
                            if model_rejected
                            else "ADAPTER_UNSUPPORTED"
                        ),
                        "matches_expected": expected_rejection,
                        "diagnostic": {
                            "code": code,
                            "message": str(error),
                        },
                    })
            records.append(record)
        summaries[dsl] = records
        (root / f"{dsl}-checks.json").write_text(json.dumps(records, indent=2, sort_keys=True))

    paired = []
    by_dsl = {
        dsl: {record["label"]: record for record in records}
        for dsl, records in summaries.items()
    }
    for label in expected:
        tilelang = by_dsl["tilelang"][label]
        triton = by_dsl["triton"][label]
        if tilelang["status"] == triton["status"] == "CHECKED":
            paired.append({
                "label": label,
                "tilelang_verdict": tilelang["actual_verdict"],
                "triton_verdict": triton["actual_verdict"],
                "matches": tilelang["actual_verdict"] == triton["actual_verdict"],
            })
    (root / "paired-checks.json").write_text(json.dumps(paired, indent=2, sort_keys=True))

    output = {}
    for dsl, records in summaries.items():
        comparable = sum(
            record["status"] in {"CHECKED", "MODEL_REJECTED"}
            for record in records
        )
        expected_matches = sum(record.get("matches_expected", False) for record in records)
        output[dsl] = {
            "statuses": dict(Counter(record["status"] for record in records)),
            "comparable": comparable,
            "expected_matches": expected_matches,
            "expected_mismatches": comparable - expected_matches,
        }
    output["paired"] = {
        "checked": len(paired),
        "matching": sum(record["matches"] for record in paired),
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
