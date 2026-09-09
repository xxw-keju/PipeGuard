from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from suite.pgir_direct.handoff_validator import validate_handoff


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
    ).stdout


def _manifest(repo: Path, artifact: Path) -> dict:
    paths = ["tracked.txt", "new.txt"]
    bundle = hashlib.sha256()
    files = []
    for name in paths:
        data = (repo / name).read_bytes()
        bundle.update(name.encode() + b"\0" + data)
        base = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{name}"],
            capture_output=True,
        )
        files.append({
            "path": name,
            "current_sha256": _sha256(data),
            "base_sha256": _sha256(base.stdout) if base.returncode == 0 else None,
            "bytes": len(data),
        })
    return {
        "schema_version": "pipeguard-direct-core-handoff-v1",
        "status": "READY_FOR_FUNCTION_LEVEL_TRANSFER",
        "source_identity": {
            "head": _git(repo, "rev-parse", "HEAD").decode().strip(),
            "tracked_binary_diff_sha256": _sha256(_git(repo, "diff", "--binary")),
            "handoff_bundle_sha256": bundle.hexdigest(),
        },
        "files": files,
        "producer_validation": {
            "full": {"collected": 3, "passed": 3, "failed": 0},
        },
        "semantic_gates": {
            "artifact": str(artifact.relative_to(repo)),
            "artifact_sha256": _sha256(artifact.read_bytes()),
        },
    }


def test_handoff_validator_checks_files_bundle_artifacts_and_git_identity(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "pipeguard@example.invalid")
    _git(repo, "config", "user.name", "PipeGuard Test")
    (repo / "tracked.txt").write_text("base\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    (repo / "tracked.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new\n")
    artifact = repo / ".artifacts" / "gate.json"
    artifact.parent.mkdir()
    artifact.write_text('{"valid":true}\n')
    manifest_path = tmp_path / "handoff.json"
    manifest_path.write_text(json.dumps(_manifest(repo, artifact)))

    valid = validate_handoff(repo, manifest_path, require_ready=True)
    assert valid["valid"] is True
    assert valid["error_count"] == 0
    assert valid["file_count"] == 2
    assert valid["artifact_count"] == 1

    (repo / "new.txt").write_text("tampered\n")
    invalid = validate_handoff(repo, manifest_path, require_ready=True)
    assert invalid["valid"] is False
    assert {error["code"] for error in invalid["errors"]} >= {
        "FILE_SHA_MISMATCH",
        "FILE_SIZE_MISMATCH",
        "BUNDLE_SHA_MISMATCH",
    }


def test_handoff_validator_rejects_nonready_and_artifact_drift(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "pipeguard@example.invalid")
    _git(repo, "config", "user.name", "PipeGuard Test")
    (repo / "tracked.txt").write_text("base\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    (repo / "new.txt").write_text("new\n")
    artifact = repo / ".artifacts" / "gate.json"
    artifact.parent.mkdir()
    artifact.write_text('{"valid":true}\n')
    manifest = _manifest(repo, artifact)
    manifest["status"] = "SUPERSEDED_PENDING_REFRESH"
    manifest_path = tmp_path / "handoff.json"
    manifest_path.write_text(json.dumps(manifest))
    artifact.write_text('{"valid":false}\n')

    result = validate_handoff(repo, manifest_path, require_ready=True)
    assert result["valid"] is False
    assert {error["code"] for error in result["errors"]} >= {
        "STATUS_NOT_READY",
        "ARTIFACT_SHA_MISMATCH",
    }
