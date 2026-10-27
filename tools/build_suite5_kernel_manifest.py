#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
INPUT_ROOT = ROOT / ".artifacts" / "suite5-inputs"
MANIFEST_SCHEMA = "pipeguard-suite5-kernel-manifest-v1"
INPUT_SCHEMA = "pipeguard-suite5-input-series-v1"
PROOF_SCHEMA = "pipeguard-wgmma-stage-union-proof-v1"
SERIES_ORDER = (
    "tilelang-gemm",
    "tilelang-attention",
    "triton-gemm",
    "triton-attention",
)
SOURCE_SPECS = (
    {
        "series_id": "tilelang-gemm",
        "path": INPUT_ROOT / "tilelang-gemm-official" / "manifest.json",
        "sha256": "319fce8ac108f95572726884823cae8e2db538bf38cc5d9f89e82588becba47d",
    },
    {
        "series_id": "tilelang-attention",
        "path": INPUT_ROOT / "tilelang-attention-noncausal" / "manifest.json",
        "sha256": "915459f711e9f8b8391d5a1192a0e701591fdb7e9c9ed8ee568883066d54179d",
    },
    {
        "series_id": "triton-gemm",
        "path": INPUT_ROOT / "triton-gemm-official" / "manifest.json",
        "sha256": "cb19d837257fda2ac756b93baf669c1a6728aabee856c244c7fd59cb7ac89bae",
    },
    {
        "series_id": "triton-attention",
        "path": INPUT_ROOT / "triton-attention-static-noncausal" / "manifest.json",
        "sha256": "a2075e4e4346aeb149feeaf915fef4ab911336d292d4bc10a4513b97625f80c1",
    },
)
ELIGIBILITY_INDEX = {
    "path": INPUT_ROOT / "wgmma-eligibility-index.json",
    "sha256": "e1655e0d3e69d12d0fe82acf91ec28c17625d3a31d6514a6b3720d7a3cb43ef0",
}


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve(manifest_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a nonempty path string")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _verify_file_digest(path: Path, expected: Any, field: str) -> None:
    if not _is_sha256(expected):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    if not path.is_file():
        raise ManifestError(f"{field} path is not a file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ManifestError(f"{field} digest mismatch: expected {expected}, got {actual}")


def _series_object(document: dict[str, Any], path: Path) -> dict[str, Any]:
    series = document.get("series")
    if isinstance(series, list):
        if len(series) != 1:
            raise ManifestError(f"source manifest must contain one series: {path}")
        series = series[0]
    if not isinstance(series, dict):
        raise ManifestError(f"source manifest series must be an object: {path}")
    return series


def _verify_proof(
    series_id: str,
    point: dict[str, Any],
    manifest_path: Path,
    eligibility_rows: dict[tuple[str, str], dict[str, Any]],
) -> None:
    if point.get("region_precision") != "EXACT_GROUP_UNION":
        raise ManifestError(f"{series_id}:{point.get('id')} lacks EXACT_GROUP_UNION")
    reference = point.get("wgmma_region_proof")
    if not isinstance(reference, dict):
        raise ManifestError(f"{series_id}:{point.get('id')} lacks WGMMA proof reference")
    if reference.get("schema_version") != PROOF_SCHEMA or reference.get("status") != "PROVED":
        raise ManifestError(f"{series_id}:{point.get('id')} has inadmissible WGMMA proof")
    proof_path = _resolve(manifest_path, reference.get("path"), "wgmma_region_proof.path")
    _verify_file_digest(proof_path, reference.get("sha256"), "wgmma_region_proof.sha256")
    proof = json.loads(proof_path.read_text())
    if proof.get("schema_version") != PROOF_SCHEMA or proof.get("status") != "PROVED":
        raise ManifestError(f"{series_id}:{point.get('id')} proof payload is not PROVED")
    scope = proof.get("scope", {})
    if scope.get("series_id") != series_id or scope.get("point_id") != point.get("id"):
        raise ManifestError(f"{series_id}:{point.get('id')} proof scope mismatch")
    bindings = proof.get("bindings", {})
    if bindings.get("snapshot", {}).get("sha256") != point.get("snapshot_sha256"):
        raise ManifestError(f"{series_id}:{point.get('id')} proof snapshot binding mismatch")
    if bindings.get("pgir", {}).get("sha256") != point.get("pgir_sha256"):
        raise ManifestError(f"{series_id}:{point.get('id')} proof PGIR binding mismatch")
    index_row = eligibility_rows.get((series_id, point["id"]))
    if index_row is None or index_row.get("proof_sha256") != reference.get("sha256"):
        raise ManifestError(f"{series_id}:{point.get('id')} eligibility index mismatch")


def build_manifest(
    source_specs: tuple[dict[str, Any], ...] = SOURCE_SPECS,
    eligibility_spec: dict[str, Any] = ELIGIBILITY_INDEX,
) -> dict[str, Any]:
    if tuple(spec["series_id"] for spec in source_specs) != SERIES_ORDER:
        raise ManifestError("source series order must match the runner contract")
    eligibility_path = Path(eligibility_spec["path"])
    _verify_file_digest(
        eligibility_path,
        eligibility_spec["sha256"],
        "wgmma_eligibility_index.sha256",
    )
    eligibility = json.loads(eligibility_path.read_text())
    if eligibility.get("status") != "PROVED" or eligibility.get("region_precision") != "EXACT_GROUP_UNION":
        raise ManifestError("WGMMA eligibility index is not PROVED")
    eligibility_rows = {
        (row["series_id"], row["point_id"]): row
        for row in eligibility.get("points", [])
        if isinstance(row, dict) and "series_id" in row and "point_id" in row
    }

    source_records = []
    output_series = []
    for spec in source_specs:
        source_path = Path(spec["path"])
        _verify_file_digest(source_path, spec["sha256"], f"{spec['series_id']}.manifest")
        document = json.loads(source_path.read_text())
        if document.get("schema_version") != INPUT_SCHEMA:
            raise ManifestError(f"{spec['series_id']} input schema mismatch")
        if document.get("status") != "READY_FOR_RECOVERY_MEASUREMENT":
            raise ManifestError(f"{spec['series_id']} is not ready for measurement")
        series = _series_object(document, source_path)
        if series.get("id") != spec["series_id"]:
            raise ManifestError(f"{spec['series_id']} source contains the wrong series")
        points = series.get("points")
        if not isinstance(points, list) or len(points) < 5:
            raise ManifestError(f"{spec['series_id']} requires at least five points")
        point_ids: set[str] = set()
        logical_events: set[int] = set()
        for point in points:
            if not isinstance(point, dict):
                raise ManifestError(f"{spec['series_id']} point must be an object")
            point_id = point.get("id")
            if not isinstance(point_id, str) or not point_id or point_id in point_ids:
                raise ManifestError(f"{spec['series_id']} has an invalid or duplicate point ID")
            point_ids.add(point_id)
            events = point.get("logical_events")
            if isinstance(events, bool) or not isinstance(events, int) or events <= 0 or events in logical_events:
                raise ManifestError(f"{spec['series_id']} has an invalid or duplicate logical-event scale")
            logical_events.add(events)
            selected_cta = point.get("selected_cta")
            if (
                not isinstance(selected_cta, list)
                or len(selected_cta) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in selected_cta)
            ):
                raise ManifestError(f"{spec['series_id']}:{point_id} has invalid selected_cta")
            if not isinstance(point.get("producer_head"), str) or not point["producer_head"]:
                raise ManifestError(f"{spec['series_id']}:{point_id} lacks producer_head")
            for path_field, digest_field in (
                ("source", "source_sha256"),
                ("snapshot", "snapshot_sha256"),
                ("pgir", "pgir_sha256"),
            ):
                artifact = _resolve(source_path, point.get(path_field), path_field)
                _verify_file_digest(artifact, point.get(digest_field), digest_field)
            if not _is_sha256(point.get("producer_dirty_diff_sha256")):
                raise ManifestError(f"{spec['series_id']}:{point_id} lacks producer diff digest")
            if spec["series_id"].startswith("tilelang-"):
                _verify_proof(spec["series_id"], point, source_path, eligibility_rows)
        source_records.append({
            "series_id": spec["series_id"],
            "path": str(source_path),
            "sha256": spec["sha256"],
            "status": document["status"],
            "points": len(points),
        })
        output_series.append(series)

    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "READY_FOR_RECOVERY_PREFLIGHT",
        "source_manifests": source_records,
        "wgmma_eligibility_index": {
            "path": str(eligibility_path),
            "sha256": eligibility_spec["sha256"],
            "status": eligibility["status"],
            "points": len(eligibility_rows),
        },
        "series": output_series,
        "summary": {
            "series": len(output_series),
            "points": sum(len(series["points"]) for series in output_series),
            "tilelang_wgmma_proved_points": len(eligibility_rows),
        },
    }


def write_new_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the proof-bound Suite 5 kernel manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    write_new_manifest(args.output, manifest)
    print(json.dumps({
        "path": str(args.output),
        "sha256": sha256_file(args.output),
        "series": manifest["summary"]["series"],
        "points": manifest["summary"]["points"],
        "tilelang_wgmma_proved_points": manifest["summary"]["tilelang_wgmma_proved_points"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
