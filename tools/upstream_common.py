from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SOURCE_INDEX_VERSION = "pipeguard-upstream-source-index-v1"
CAPSULE_MANIFEST_VERSION = "pipeguard-upstream-capsule-v1"
RECIPE_VERSION = "pipeguard-upstream-recipe-v1"
RUN_RESULT_VERSION = "pipeguard-upstream-run-v1"
TOOL_VERSION = "pipeguard-upstream-tools-v1"

DSL_LOCKS = {
    "tilelang": {
        "commit": "2c31e071c626c0498b48dcdc2741dc8651b70dc1",
        "backend_commit": "907a88c8791ccf33b9874821bc875e7abf624367",
        "environment": "tl-pg",
        "snapshot_version": "tilelang-snapshot-v2",
    },
    "triton": {
        "commit": "61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f",
        "backend_commit": None,
        "environment": "triton-pg",
        "snapshot_version": "triton-snapshot-v2",
    },
}

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def stable_id(*parts: Any, length: int = 16) -> str:
    return sha256_text(canonical_json(parts))[:length]


def safe_component(value: str, *, maximum: int = 80) -> str:
    normalized = _SAFE_COMPONENT.sub("_", value).strip("._") or "anonymous"
    if len(normalized) <= maximum:
        return normalized
    return normalized[: maximum - 9] + "__" + sha256_text(normalized)[:7]


def git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cannot read Git commit for {root}: {process.stderr.strip()}")
    return process.stdout.strip()


def git_blob_hash(root: Path, relative_path: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "hash-object", str(relative_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"cannot hash tracked source {relative_path}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def require_locked_checkout(dsl: str, root: Path, *, allow_commit_mismatch: bool = False) -> str:
    resolved = root.resolve(strict=True)
    actual = git_commit(resolved)
    expected = DSL_LOCKS[dsl]["commit"]
    if actual != expected and not allow_commit_mismatch:
        raise RuntimeError(
            f"{dsl} checkout mismatch: expected {expected}, got {actual} at {resolved}"
        )
    return actual


def path_beneath(root: Path, path: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"path escapes locked source root: {resolved_path}") from error


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
