from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TOOLS))

import triton_capture  # noqa: E402
from suite.frontend_pairs.catalog import PAIRS  # noqa: E402
from suite.frontend_pairs.sources.triton_cases import compile_triton_case  # noqa: E402


def _binding_evidence(raw_path: Path, family: str) -> dict:
    pointer_allocations = {}
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF", "MEMORY", "CTA-BARRIER"}:
        pointer_allocations = {"0": "global:data", "1": "global:output"}
    elif family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-LOAD"}:
        # Gluon expands the descriptor into arguments 0..4; the ordinary
        # input/output pointers follow at positions 5 and 6.
        pointer_allocations = {"5": "global:input", "6": "global:output"}
    elif family == "TMA-STORE":
        pointer_allocations = {"5": "global:output", "6": "global:sink"}
    elif family == "PROXY-STORE":
        pointer_allocations = {
            "5": "global:output",
            "6": "global:sink",
        }
    return {
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": pointer_allocations,
        "scalar_values": {},
    }


MAX_BINDING_EVIDENCE_ATTEMPTS = 64


def _capture_with_evidence(
    raw_path: Path,
    evidence_path: Path,
    family: str,
    label: str,
    *,
    descriptor_shape: list[int] | None = None,
    final_ptx_path: Path | None = None,
) -> dict:
    evidence = _binding_evidence(raw_path, family)
    for _attempt in range(MAX_BINDING_EVIDENCE_ATTEMPTS):
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        try:
            return triton_capture.capture_file(
                raw_path,
                source=f"suite2:{label}",
                command="suite2-native-triton-compile-only",
                grid_ctas=1,
                binding_evidence_path=evidence_path,
                final_ptx_path=final_ptx_path,
            )
        except triton_capture.CaptureError as error:
            operation_index = error.details.get("operation_index")
            if not isinstance(operation_index, int):
                raise
            if error.code == "TRITON_BINDING_EVIDENCE_REQUIRED":
                if family == "PROXY-STORE" and "warp_specialize" in str(error):
                    evidence["array_attributes"][str(operation_index)] = {
                        "partition_num_warps": [4],
                        "warp_group_start_ids": [4],
                    }
                else:
                    evidence["unit_attributes"][str(operation_index)] = []
                continue
            if error.code != "TRITON_TMA_DESCRIPTOR_UNRESOLVED":
                raise
            is_named_store = family == "TMA-STORE"
            is_proxy_store = family == "PROXY-STORE"
            allocation = (
                "global:output"
                if family in {"TMA-STORE", "TMA-STORE-ENDPOINT", "PROXY-STORE"}
                else "global:input"
            )
            shape = descriptor_shape or ([64, 32] if is_named_store else [32, 32])
            evidence["array_attributes"][str(operation_index)] = {
                "descriptor": {
                    "allocation_id": allocation,
                    "coordinate_order": [0, 1],
                    "element_bytes": 2,
                    "shape": shape,
                    "strides": [shape[1], 1],
                    "tile_shape": [64, 64] if is_proxy_store else [32, 32],
                }
            }
    raise AssertionError("descriptor evidence retry did not return or raise")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", action="append", type=int)
    parser.add_argument("--family", action="append")
    parser.add_argument("--label", action="append")
    args = parser.parse_args()

    output = Path(args.output)
    raw_root = output / "raw" / "triton"
    snapshot_root = output / "snapshots" / "triton"
    evidence_root = output / "evidence" / "triton"
    result_path = output / "triton-results.json"
    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
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
        raw_path = raw_root / f"{case.label}.ttgir.mlir"
        snapshot_path = snapshot_root / f"{case.label}.json"
        evidence_path = evidence_root / f"{case.label}.json"
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
            raw_path.write_text(compile_triton_case(case.family, case.scenario_id, case.variant, case.polarity))
            snapshot = _capture_with_evidence(
                raw_path,
                evidence_path,
                case.family,
                case.label,
            )
            snapshot_path.write_text(json.dumps(snapshot, sort_keys=True))
            record.update({
                "status": "CAPTURED",
                "raw_path": str(raw_path.relative_to(output)),
                "snapshot_path": str(snapshot_path.relative_to(output)),
                "operations": len(snapshot["operations"]),
            })
        except Exception as error:
            diagnostic = error.diagnostic() if isinstance(error, triton_capture.CaptureError) else {
                "code": "TRITON_NATIVE_LOWERING_FAILED",
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
        "dsl": "triton",
        "processed": processed,
        "recorded": len(records_by_label),
        "counts": counts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
