from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from upstream_common import canonical_json, sha256_bytes


SUITE_LOCK_VERSION = "pipeguard-upstream-suite-lock-v1"
KERNEL_CATALOG_VERSION = "pipeguard-upstream-kernel-catalog-v1"
CHECKER_RESULT_VERSION = "pipeguard-upstream-checker-result-v1"


class SuiteValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SuiteSummary:
    candidates: int
    admitted_kernels: int
    original_specs: int
    boundary_specs: int
    fixed_controls: int
    intentional_invalid_specs: int


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _fail(message: str) -> None:
    raise SuiteValidationError(message)


def _relative_file(root: Path, text: str, *, label: str) -> Path:
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{label} escapes corpus root: {text}")
    path = root / relative
    if not path.is_file():
        _fail(f"{label} does not exist: {text}")
    return path


def _unique(items: Iterable[dict[str, Any]], key: str, *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item[key]
        if value in indexed:
            _fail(f"duplicate {label} {value}")
        indexed[value] = item
    return indexed


def _schema_validate(document: Any, schema_path: Path) -> None:
    schema = _load(schema_path)
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as error:
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        _fail(f"{schema_path.name} rejected {location}: {error.message}")


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        _fail(f"invalid JSON pointer {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _pointer_get(value: Any, pointer: str) -> Any:
    current = value
    for part in _pointer_parts(pointer):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as error:
                _fail(f"JSON pointer {pointer} does not resolve")
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            _fail(f"JSON pointer {pointer} does not resolve")
    return current


def _diff_paths(left: Any, right: Any, pointer: str = "") -> set[str]:
    if type(left) is not type(right):
        return {pointer}
    if isinstance(left, dict):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            child = f"{pointer}/{_pointer_escape(key)}"
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return {pointer}
        paths: set[str] = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.update(_diff_paths(left_item, right_item, f"{pointer}/{index}"))
        return paths
    return set() if left == right else {pointer}


def _normalized_recipe_for_delta(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(recipe))
    normalized.pop("recipe_id", None)
    normalized.pop("origin", None)
    return normalized


def _grid(recipe: dict[str, Any]) -> tuple[int, int, int] | None:
    compiler = recipe.get("compiler", {})
    raw = compiler.get("grid")
    if raw is None:
        count = compiler.get("grid_ctas")
        return (count, 1, 1) if isinstance(count, int) else None
    values = list(raw) + [1] * (3 - len(raw))
    return tuple(values[:3])


def _validate_lock(corpus_root: Path, lock: dict[str, Any], lock_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema_root = corpus_root / "schemas"
    _schema_validate(lock, schema_root / "suite-lock-v1.schema.json")
    if lock["schema_version"] != SUITE_LOCK_VERSION:
        _fail(f"unsupported suite lock version {lock['schema_version']}")

    repositories = _unique(lock["repositories"], "source_id", label="repository source_id")
    toolchains = _unique(lock["toolchains"], "toolchain_id", label="toolchain_id")
    profiles = _unique(lock["target_profiles"], "profile_id", label="target profile")

    baseline = lock["normative_baseline"]
    baseline_path = _relative_file(corpus_root, baseline["source"], label="normative baseline")
    if _hash(baseline_path) != baseline["sha256"]:
        _fail("normative baseline hash mismatch")
    baseline_document = _load(baseline_path)
    if baseline_document.get("baseline_id") != baseline["document_id"]:
        _fail("normative baseline document_id mismatch")
    _unique(baseline_document.get("documents", []), "source_id", label="normative source")
    _unique(baseline_document.get("rules", []), "rule_id", label="normative rule")

    for source_id, repository in repositories.items():
        index_path = _relative_file(corpus_root, repository["source_index"]["path"], label=f"{source_id} source index")
        if _hash(index_path) != repository["source_index"]["sha256"]:
            _fail(f"{source_id} source index hash mismatch")
        index = _load(index_path)
        upstream = index.get("upstream", {})
        if upstream.get("expected_commit") != repository["commit"]:
            _fail(f"{source_id} source index commit mismatch")
        if index.get("include_prefixes") != repository["source_prefixes"]:
            _fail(f"{source_id} include prefixes differ from source index")
        if index.get("dsl") != repository["dsl"]:
            _fail(f"{source_id} DSL differs from source index")
        audit_kinds: set[str] = set()
        for audit in repository.get("audit_documents", []):
            if audit["kind"] in audit_kinds:
                _fail(f"{source_id} repeats audit kind {audit['kind']}")
            audit_kinds.add(audit["kind"])
            audit_path = _relative_file(
                corpus_root,
                audit["path"],
                label=f"{source_id} {audit['kind'].lower()}",
            )
            if _hash(audit_path) != audit["sha256"]:
                _fail(f"{source_id} {audit['kind']} hash mismatch")

    for toolchain_id, toolchain in toolchains.items():
        repository = repositories.get(toolchain["frontend_source_id"])
        if repository is None:
            _fail(f"{toolchain_id} references unknown frontend source")
        if toolchain["frontend_commit"] != repository["commit"]:
            _fail(f"{toolchain_id} frontend commit differs from source lock")

    return repositories, toolchains, profiles


def _validate_evidence(
    evidence: dict[str, dict[str, Any]],
    repositories: dict[str, dict[str, Any]],
) -> None:
    for evidence_id, item in evidence.items():
        start = item.get("start_line")
        end = item.get("end_line")
        if (start is None) != (end is None):
            _fail(f"{evidence_id} must provide both start_line and end_line")
        if start is not None and start > end:
            _fail(f"{evidence_id} has an inverted line range")
        repository_id = item.get("repository_id")
        if repository_id is None:
            continue
        repository = repositories.get(repository_id)
        if repository is None:
            _fail(f"{evidence_id} references unknown repository {repository_id}")
        if item.get("commit") != repository["commit"]:
            _fail(f"{evidence_id} commit differs from repository lock")
        if repository["commit"] not in item["url"]:
            _fail(f"{evidence_id} URL is not commit-qualified")


def _candidate_lookup(corpus_root: Path, repositories: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for source_id, repository in repositories.items():
        index_path = _relative_file(corpus_root, repository["source_index"]["path"], label=f"{source_id} source index")
        index = _load(index_path)
        for candidate in index["candidates"]:
            candidates[(source_id, candidate["candidate_id"])] = candidate
    return candidates


def _validate_revisions(
    corpus_root: Path,
    revisions: dict[str, dict[str, Any]],
    kernels: dict[str, dict[str, Any]],
    repositories: dict[str, dict[str, Any]],
    toolchains: dict[str, dict[str, Any]],
) -> None:
    candidates = _candidate_lookup(corpus_root, repositories)
    for revision_id, revision in revisions.items():
        kernel = kernels.get(revision["kernel_id"])
        if kernel is None:
            _fail(f"{revision_id} references unknown kernel")
        repository = repositories.get(revision["repository_id"])
        if repository is None:
            _fail(f"{revision_id} references unknown repository")
        toolchain_id = revision["toolchain_id"]
        if toolchain_id is not None and toolchain_id not in toolchains:
            _fail(f"{revision_id} references unknown toolchain")
        if revision["commit"] != repository["commit"]:
            _fail(f"{revision_id} commit differs from source lock")
        candidate = candidates.get((revision["repository_id"], revision["candidate_id"]))
        if candidate is None:
            _fail(f"{revision_id} candidate is absent from source index")
        for field, candidate_field in (
            ("path", "relative_path"),
            ("qualname", "qualname"),
            ("start_line", "start_line"),
            ("end_line", "end_line"),
        ):
            if revision[field] != candidate[candidate_field]:
                _fail(f"{revision_id} {field} differs from source index")
        manifest_path = _relative_file(corpus_root, candidate["manifest_path"], label=f"{revision_id} manifest")
        manifest = _load(manifest_path)
        upstream = manifest["upstream"]
        for field in ("commit", "git_blob", "source_sha256", "selected_source_sha256"):
            if revision[field] != upstream[field]:
                _fail(f"{revision_id} {field} differs from manifest")
        if revision["start_line"] > revision["end_line"]:
            _fail(f"{revision_id} has an inverted line range")
        if revision["commit"] not in revision["permalink"]:
            _fail(f"{revision_id} permalink is not commit-qualified")
        anchor = f"#L{revision['start_line']}-L{revision['end_line']}"
        if anchor not in revision["permalink"]:
            _fail(f"{revision_id} permalink line anchor mismatch")


def _load_recipe(corpus_root: Path, spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    reference = spec["recipe"]
    path = _relative_file(corpus_root, reference["path"], label=f"{spec['spec_id']} recipe")
    if _hash(path) != reference["sha256"]:
        _fail(f"{spec['spec_id']} recipe hash mismatch")
    recipe = _load(path)
    try:
        jsonschema.validate(recipe, schema)
    except jsonschema.ValidationError as error:
        _fail(f"{spec['spec_id']} recipe schema failure: {error.message}")
    if recipe["recipe_id"] != reference["recipe_id"]:
        _fail(f"{spec['spec_id']} recipe_id mismatch")
    if recipe["candidate_id"] != reference["candidate_id"]:
        _fail(f"{spec['spec_id']} candidate_id mismatch")
    return recipe


def _validate_specs(
    corpus_root: Path,
    lock: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    revisions: dict[str, dict[str, Any]],
    kernels: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    normative_rules: set[str],
) -> dict[str, dict[str, Any]]:
    recipe_schema = _load(corpus_root / "schemas" / "recipe-v1.schema.json")
    recipes = {spec_id: _load_recipe(corpus_root, spec, recipe_schema) for spec_id, spec in specs.items()}

    for spec_id, spec in specs.items():
        kernel = kernels.get(spec["kernel_id"])
        revision = revisions.get(spec["revision_id"])
        if kernel is None or revision is None:
            _fail(f"{spec_id} references an unknown kernel or revision")
        if revision["kernel_id"] != spec["kernel_id"]:
            _fail(f"{spec_id} revision belongs to a different kernel")
        if spec["recipe"]["candidate_id"] != revision["candidate_id"]:
            _fail(f"{spec_id} recipe candidate differs from revision")
        unknown_evidence = set(spec["evidence_ids"]) - set(evidence)
        unknown_ground_truth = set(spec["semantic_ground_truth"]["evidence_ids"]) - set(evidence)
        if unknown_evidence or unknown_ground_truth:
            _fail(f"{spec_id} references unknown evidence")
        unknown_rules = set(spec["rule_ids"]) - normative_rules
        if unknown_rules:
            _fail(f"{spec_id} references unknown rules {sorted(unknown_rules)}")

        role = spec["role"]
        parent_id = spec.get("parent_spec_id")
        if role == "ORIGINAL":
            if parent_id is not None or spec["delta"] or spec["boundary_dimensions"]:
                _fail(f"{spec_id} ORIGINAL cannot have parent, delta, or boundary dimensions")
        else:
            if parent_id not in specs:
                _fail(f"{spec_id} requires a valid parent_spec_id")
            parent = specs[parent_id]
            if parent["kernel_id"] != spec["kernel_id"]:
                _fail(f"{spec_id} parent belongs to another kernel")
            if role == "BOUNDARY":
                if not spec["delta"] or not spec["boundary_dimensions"] or not spec.get("boundary_transition"):
                    _fail(f"{spec_id} BOUNDARY requires delta, dimensions, and transition")
            parent_recipe = _normalized_recipe_for_delta(recipes[parent_id])
            recipe = _normalized_recipe_for_delta(recipes[spec_id])
            actual_paths = _diff_paths(parent_recipe, recipe)
            declared_paths = {item["path"] for item in spec["delta"]}
            if actual_paths != declared_paths:
                _fail(f"{spec_id} declared delta {sorted(declared_paths)} differs from recipe delta {sorted(actual_paths)}")
            for delta in spec["delta"]:
                if _pointer_get(parent_recipe, delta["path"]) != delta["before"]:
                    _fail(f"{spec_id} delta before-value mismatch at {delta['path']}")
                if _pointer_get(recipe, delta["path"]) != delta["after"]:
                    _fail(f"{spec_id} delta after-value mismatch at {delta['path']}")

        pipeline_status = spec["expected_pipeline"]["status"]
        expected_checker = spec.get("expected_checker")
        if pipeline_status == "CAPTURED" and revision["toolchain_id"] is None:
            _fail(f"{spec_id} CAPTURED case requires a locked toolchain")
        if pipeline_status == "CAPTURED" and expected_checker is None:
            _fail(f"{spec_id} CAPTURED case requires expected_checker")
        if pipeline_status != "CAPTURED" and expected_checker is not None:
            _fail(f"{spec_id} non-captured case cannot declare checker verdict")
        if role == "INTENTIONAL_INVALID" and expected_checker and expected_checker["verdict"] == "SAFE":
            _fail(f"{spec_id} intentional invalid case cannot expect SAFE")

        ground_truth = spec["semantic_ground_truth"]
        if ground_truth["status"] == "UNADJUDICATED":
            if ground_truth["class"] != "UNKNOWN" or ground_truth["method"] != "UNADJUDICATED":
                _fail(f"{spec_id} unadjudicated ground truth must remain UNKNOWN")
        elif ground_truth["method"] == "UNADJUDICATED":
            _fail(f"{spec_id} adjudicated ground truth needs independent method")

        execution = spec["execution"]
        symbols = execution["symbols"]
        if execution["symbol_policy"] == "EXACT" and not symbols:
            _fail(f"{spec_id} EXACT symbol policy requires named symbols")
        if execution["symbol_policy"] in {"ONLY", "ALL"} and symbols:
            _fail(f"{spec_id} {execution['symbol_policy']} symbol policy cannot list symbols")
        grid = _grid(recipes[spec_id])
        selected = tuple(execution["selected_cta"])
        if grid is not None and any(value >= bound for value, bound in zip(selected, grid)):
            _fail(f"{spec_id} selected CTA {selected} lies outside grid {grid}")
        if execution["claim_scope"] == "SINGLE_CTA" and grid is not None and grid != (1, 1, 1):
            _fail(f"{spec_id} SINGLE_CTA claim has multi-CTA grid {grid}")

    policy = lock["policy"]
    for kernel_id, kernel in kernels.items():
        kernel_specs = [spec for spec in specs.values() if spec["kernel_id"] == kernel_id]
        original = [spec for spec in kernel_specs if spec["role"] == "ORIGINAL"]
        boundaries = [spec for spec in kernel_specs if spec["role"] == "BOUNDARY"]
        if kernel["admission"] == "CANDIDATE":
            if not kernel.get("exclusion_reason"):
                _fail(f"candidate kernel {kernel_id} requires exclusion_reason")
            continue
        if kernel.get("exclusion_reason"):
            _fail(f"admitted kernel {kernel_id} cannot have exclusion_reason")
        if len(original) < policy["minimum_original_specs"]:
            _fail(f"admitted kernel {kernel_id} lacks original specifications")
        if len(boundaries) < policy["minimum_boundary_specs"]:
            _fail(f"admitted kernel {kernel_id} lacks boundary specifications")
        dimensions = {dimension for spec in boundaries for dimension in spec["boundary_dimensions"]}
        if len(dimensions) < policy["minimum_boundary_dimensions"]:
            _fail(f"admitted kernel {kernel_id} lacks independent boundary dimensions")
        primary = specs.get(kernel.get("primary_original_spec_id"))
        if primary is None or primary["kernel_id"] != kernel_id or primary["role"] != "ORIGINAL":
            _fail(f"admitted kernel {kernel_id} has invalid primary original specification")

    return recipes


def validate_suite(
    corpus_root: Path,
    *,
    lock_path: Path | None = None,
    catalog_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], SuiteSummary]:
    corpus_root = corpus_root.resolve(strict=True)
    lock_path = (lock_path or corpus_root / "suite3" / "lock.json").resolve(strict=True)
    catalog_path = (catalog_path or corpus_root / "suite3" / "catalog.json").resolve(strict=True)
    lock = _load(lock_path)
    catalog = _load(catalog_path)

    repositories, toolchains, _profiles = _validate_lock(corpus_root, lock, lock_path)
    _schema_validate(catalog, corpus_root / "schemas" / "kernel-catalog-v1.schema.json")
    if catalog["schema_version"] != KERNEL_CATALOG_VERSION:
        _fail(f"unsupported catalog version {catalog['schema_version']}")
    if catalog["suite_id"] != lock["suite_id"]:
        _fail("catalog suite_id differs from lock")
    if catalog["suite_lock_sha256"] != _hash(lock_path):
        _fail("catalog suite lock hash mismatch")

    evidence = _unique(catalog["source_evidence"], "evidence_id", label="evidence_id")
    revisions = _unique(catalog["kernel_revisions"], "revision_id", label="revision_id")
    kernels = _unique(catalog["kernels"], "kernel_id", label="kernel_id")
    specs = _unique(catalog["specifications"], "spec_id", label="spec_id")
    _validate_evidence(evidence, repositories)
    _validate_revisions(corpus_root, revisions, kernels, repositories, toolchains)

    baseline_path = _relative_file(corpus_root, lock["normative_baseline"]["source"], label="normative baseline")
    normative_rules = {rule["rule_id"] for rule in _load(baseline_path)["rules"]}
    _validate_specs(corpus_root, lock, evidence, revisions, kernels, specs, normative_rules)

    for kernel_id, kernel in kernels.items():
        for revision_id in kernel["revision_ids"]:
            revision = revisions.get(revision_id)
            if revision is None or revision["kernel_id"] != kernel_id:
                _fail(f"{kernel_id} references invalid revision {revision_id}")

    role_counts: dict[str, int] = {}
    for spec in specs.values():
        role_counts[spec["role"]] = role_counts.get(spec["role"], 0) + 1
    summary = SuiteSummary(
        candidates=sum(kernel["admission"] == "CANDIDATE" for kernel in kernels.values()),
        admitted_kernels=sum(kernel["admission"] == "ADMITTED" for kernel in kernels.values()),
        original_specs=role_counts.get("ORIGINAL", 0),
        boundary_specs=role_counts.get("BOUNDARY", 0),
        fixed_controls=role_counts.get("FIXED_CONTROL", 0),
        intentional_invalid_specs=role_counts.get("INTENTIONAL_INVALID", 0),
    )
    return lock, catalog, summary


def validate_checker_result(corpus_root: Path, result: dict[str, Any]) -> None:
    corpus_root = corpus_root.resolve(strict=True)
    _schema_validate(result, corpus_root / "schemas" / "checker-result-v1.schema.json")
    if result["schema_version"] != CHECKER_RESULT_VERSION:
        _fail(f"unsupported checker result version {result['schema_version']}")
    checked = result["status"] == "CHECKED"
    if checked and ("pgir" not in result or "actual" not in result):
        _fail("CHECKED result requires PGIR and actual checker output")
    if not checked and "actual" in result:
        _fail(f"{result['status']} result cannot contain checker verdict")
    if result["status"] == "CAPTURE_NOT_AVAILABLE" and "pgir" in result:
        _fail("CAPTURE_NOT_AVAILABLE result cannot identify a checked PGIR")
