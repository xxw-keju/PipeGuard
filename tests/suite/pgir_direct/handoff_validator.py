from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=check,
    )


def _artifact_entries(value: Any) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(value, dict):
        artifact = value.get("artifact")
        digest = value.get("artifact_sha256")
        if isinstance(artifact, str) and isinstance(digest, str):
            entries.append((artifact, digest))
        for item in value.values():
            entries.extend(_artifact_entries(item))
    elif isinstance(value, list):
        for item in value:
            entries.extend(_artifact_entries(item))
    return entries


def validate_handoff(repo: Path, manifest_path: Path, *, require_ready: bool) -> dict[str, Any]:
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded)
    errors: list[dict[str, Any]] = []

    def error(code: str, message: str, **details: Any) -> None:
        errors.append({"code": code, "message": message, **details})

    if manifest.get("schema_version") != "pipeguard-direct-core-handoff-v1":
        error("SCHEMA_VERSION", "handoff schema version is unsupported")
    if require_ready and manifest.get("status") != "READY_FOR_FUNCTION_LEVEL_TRANSFER":
        error("STATUS_NOT_READY", "handoff is not ready for transfer")

    source = manifest.get("source_identity", {})
    actual_head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    if source.get("head") != actual_head:
        error("HEAD_MISMATCH", "producer HEAD does not match manifest", expected=source.get("head"), actual=actual_head)
    tracked_diff = _git(repo, "diff", "--binary").stdout
    actual_diff = _sha256(tracked_diff)
    if source.get("tracked_binary_diff_sha256") != actual_diff:
        error(
            "TRACKED_DIFF_MISMATCH",
            "tracked binary diff digest does not match manifest",
            expected=source.get("tracked_binary_diff_sha256"),
            actual=actual_diff,
        )

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        error("FILES_MISSING", "handoff has no transfer file entries")
        rows = []
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(paths) != len(set(paths)):
        error("FILE_DUPLICATE", "handoff transfer file paths are not unique")
    bundle = hashlib.sha256()
    for row in rows:
        path_text = row.get("path")
        if not isinstance(path_text, str):
            error("FILE_PATH_INVALID", "handoff file path is not a string")
            continue
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts:
            error("FILE_PATH_INVALID", "handoff file path escapes the repository", path=path_text)
            continue
        path = repo / relative
        try:
            data = path.read_bytes()
        except OSError as caught:
            error("FILE_UNREADABLE", "handoff file cannot be read", path=path_text, detail=str(caught))
            continue
        bundle.update(path_text.encode() + b"\0" + data)
        actual_sha = _sha256(data)
        if row.get("current_sha256") != actual_sha:
            error("FILE_SHA_MISMATCH", "handoff file digest differs", path=path_text, expected=row.get("current_sha256"), actual=actual_sha)
        if row.get("bytes") != len(data):
            error("FILE_SIZE_MISMATCH", "handoff file size differs", path=path_text, expected=row.get("bytes"), actual=len(data))
        base = _git(repo, "show", f"HEAD:{path_text}", check=False)
        actual_base = _sha256(base.stdout) if base.returncode == 0 else None
        if row.get("base_sha256") != actual_base:
            error("BASE_SHA_MISMATCH", "handoff base digest differs", path=path_text, expected=row.get("base_sha256"), actual=actual_base)
    actual_bundle = bundle.hexdigest()
    if source.get("handoff_bundle_sha256") != actual_bundle:
        error("BUNDLE_SHA_MISMATCH", "handoff bundle digest differs", expected=source.get("handoff_bundle_sha256"), actual=actual_bundle)

    seen_artifacts: set[str] = set()
    for artifact_text, expected_sha in _artifact_entries(manifest):
        if artifact_text in seen_artifacts:
            continue
        seen_artifacts.add(artifact_text)
        relative = Path(artifact_text)
        if relative.is_absolute() or ".." in relative.parts:
            error("ARTIFACT_PATH_INVALID", "artifact path escapes the repository", path=artifact_text)
            continue
        artifact = repo / relative
        try:
            actual_sha = _sha256(artifact.read_bytes())
        except OSError as caught:
            error("ARTIFACT_UNREADABLE", "handoff artifact cannot be read", path=artifact_text, detail=str(caught))
            continue
        if actual_sha != expected_sha:
            error("ARTIFACT_SHA_MISMATCH", "handoff artifact digest differs", path=artifact_text, expected=expected_sha, actual=actual_sha)

    validation = manifest.get("producer_validation", {}).get("full", {})
    if (
        validation.get("failed") != 0
        or validation.get("collected") != validation.get("passed")
        or not isinstance(validation.get("passed"), int)
    ):
        error("PRODUCER_VALIDATION_INVALID", "full producer validation is not all-pass")

    return {
        "schema_version": "pipeguard-direct-core-handoff-validation-v1",
        "valid": not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(encoded),
        "repo": str(repo),
        "status": manifest.get("status"),
        "file_count": len(rows),
        "artifact_count": len(seen_artifacts),
        "computed_bundle_sha256": actual_bundle,
        "computed_tracked_diff_sha256": actual_diff,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    result = validate_handoff(
        Path(args.repo),
        Path(args.manifest),
        require_ready=args.require_ready,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
