from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema


LOCK_VERSION = "pipeguard-mutation-suite-lock-v1"
CATALOG_VERSION = "pipeguard-mutation-catalog-v1"
RULES_VERSION = "pipeguard-mutation-rules-v1"
WORKLOAD_VERSION = "pipeguard-mutation-workload-v1"
BASELINE_VERSION = "pipeguard-mutation-baseline-index-v1"


class MutationSuiteValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MutationSuiteSummary:
    kernels: int
    original_variants: int
    mutant_variants: int
    original_workloads: int
    boundary_workloads: int
    cases: int


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fail(message: str) -> None:
    raise MutationSuiteValidationError(message)


def _relative_file(repository_root: Path, path_text: str, *, label: str) -> Path:
    path = (repository_root / path_text).resolve(strict=True)
    try:
        path.relative_to(repository_root)
    except ValueError:
        _fail(f"{label} escapes repository root: {path_text}")
    if not path.is_file():
        _fail(f"{label} is not a file: {path_text}")
    return path


def _unique(items: list[dict[str, Any]], key: str, *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item[key]
        if value in result:
            _fail(f"duplicate {label}: {value}")
        result[value] = item
    return result


def _schema_validate(value: dict[str, Any], path: Path) -> None:
    try:
        jsonschema.validate(value, load_json(path))
    except jsonschema.ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path)
        _fail(f"{path.name} validation failed at {location or '<root>'}: {error.message}")


def _diff_bindings(left: Any, right: Any, pointer: str = "/bindings") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": pointer, "before": left, "after": right}]
    if isinstance(left, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
            if key not in left:
                result.append({"path": child, "before": {"__pipeguard_missing__": True}, "after": right[key]})
            elif key not in right:
                result.append({"path": child, "before": left[key], "after": {"__pipeguard_missing__": True}})
            else:
                result.extend(_diff_bindings(left[key], right[key], child))
        return result
    if isinstance(left, list):
        if left == right:
            return []
        return [{"path": pointer, "before": left, "after": right}]
    return [] if left == right else [{"path": pointer, "before": left, "after": right}]


def validate_mutation_suite(
    repository_root: Path,
    *,
    lock_path: Path | None = None,
    catalog_path: Path | None = None,
    baseline_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], MutationSuiteSummary]:
    repository_root = repository_root.resolve(strict=True)
    suite_root = repository_root / "corpus" / "mutations" / "suite4"
    schema_root = repository_root / "corpus" / "mutations" / "schemas"
    lock_path = (lock_path or suite_root / "lock.json").resolve(strict=True)
    catalog_path = (catalog_path or suite_root / "catalog.json").resolve(strict=True)
    baseline_path = (baseline_path or suite_root / "baseline-index.json").resolve(strict=True)

    lock = load_json(lock_path)
    catalog = load_json(catalog_path)
    baseline = load_json(baseline_path)
    _schema_validate(lock, schema_root / "mutation-suite-lock-v1.schema.json")
    _schema_validate(catalog, schema_root / "mutation-catalog-v1.schema.json")
    _schema_validate(baseline, schema_root / "mutation-baseline-index-v1.schema.json")
    if lock["schema_version"] != LOCK_VERSION or catalog["schema_version"] != CATALOG_VERSION:
        _fail("unsupported Suite 4 schema version")
    if baseline["schema_version"] != BASELINE_VERSION:
        _fail("unsupported Suite 4 baseline schema version")
    if len(catalog["kernels"]) != lock["policy"]["kernel_count"]:
        _fail("catalog kernel count differs from locked policy")
    if catalog["suite_id"] != lock["suite_id"] or baseline["suite_id"] != lock["suite_id"]:
        _fail("Suite 4 identifiers differ")
    lock_hash = sha256_file(lock_path)
    if catalog["suite_lock_sha256"] != lock_hash or baseline["suite_lock_sha256"] != lock_hash:
        _fail("Suite 4 lock hash mismatch")
    if baseline["catalog_sha256"] != sha256_file(catalog_path):
        _fail("Suite 4 catalog hash mismatch")

    for ref_name in ("lock", "catalog"):
        ref = lock["suite3"][ref_name]
        path = _relative_file(repository_root, ref["path"], label=f"Suite 3 {ref_name}")
        if sha256_file(path) != ref["sha256"]:
            _fail(f"Suite 3 {ref_name} hash mismatch")
    rules_ref = lock["mutation_rules"]
    rules_path = _relative_file(repository_root, rules_ref["path"], label="mutation rules")
    if sha256_file(rules_path) != rules_ref["sha256"]:
        _fail("mutation rule hash mismatch")
    rules = load_json(rules_path)
    _schema_validate(rules, schema_root / "mutation-rules-v1.schema.json")
    if rules["schema_version"] != RULES_VERSION:
        _fail("unsupported mutation rule schema version")
    rule_map = _unique(rules["rules"], "rule_id", label="mutation rule")

    repository_map = _unique(lock["repositories"], "source_id", label="source repository")
    toolchains = _unique(lock["toolchains"], "toolchain_id", label="toolchain")
    for toolchain_id, item in toolchains.items():
        source = repository_map.get(item["frontend_source_id"])
        if source is None or source["commit"] != item["frontend_commit"] or source["dsl"] != item["dsl"]:
            _fail(f"{toolchain_id} frontend identity mismatch")
    for source_id, item in repository_map.items():
        path = _relative_file(repository_root, item["source_index"]["path"], label=f"{source_id} index")
        if sha256_file(path) != item["source_index"]["sha256"]:
            _fail(f"{source_id} source index hash mismatch")
        index = load_json(path)
        if index["upstream"]["expected_commit"] != item["commit"]:
            _fail(f"{source_id} source index commit mismatch")

    kernels = _unique(catalog["kernels"], "kernel_id", label="kernel")
    variants = _unique(catalog["variants"], "variant_id", label="variant")
    workloads = _unique(catalog["workloads"], "spec_id", label="workload")
    cases = _unique(catalog["cases"], "case_id", label="case")

    families: dict[str, list[dict[str, Any]]] = {}
    for kernel in kernels.values():
        families.setdefault(kernel["operator_family"], []).append(kernel)
    if len(families) != lock["policy"]["operator_family_count"]:
        _fail("catalog operator-family count differs from locked policy")
    for family, implementations in families.items():
        if len(implementations) != lock["policy"]["implementations_per_family"]:
            _fail(f"{family} lacks the required paired implementations")
        if {item["dsl"] for item in implementations} != {"triton", "tilelang"}:
            _fail(f"{family} must contain one Triton and one TileLang implementation")
        if any(item["project"] not in {"Triton", "TileLang"} for item in implementations):
            _fail(f"{family} contains a non-official project")

    for kernel_id, kernel in kernels.items():
        if kernel["source_id"] not in repository_map:
            _fail(f"{kernel_id} references unknown source repository")
        kernel_variants = [variants[item] for item in kernel["variant_ids"]]
        kernel_workloads = [workloads[item] for item in kernel["workload_spec_ids"]]
        if any(item["kernel_id"] != kernel_id for item in kernel_variants + kernel_workloads):
            _fail(f"{kernel_id} references a foreign variant or workload")
        originals = [item for item in kernel_variants if item["role"] == "ORIGINAL"]
        mutants = [item for item in kernel_variants if item["role"] == "MUTANT"]
        if len(originals) != 1:
            _fail(f"{kernel_id} must have exactly one original source variant")
        if len(mutants) < lock["policy"]["minimum_mutants"]:
            _fail(f"{kernel_id} lacks required source mutants")
        families = {item["mutation_family"] for item in mutants}
        if len(families) < lock["policy"]["minimum_mutation_families"]:
            _fail(f"{kernel_id} lacks mutation-family coverage")
        original_id = originals[0]["variant_id"]
        for item in kernel_variants:
            source_path = _relative_file(repository_root, item["source"]["path"], label=item["variant_id"])
            if sha256_file(source_path) != item["source"]["sha256"]:
                _fail(f"{item['variant_id']} source hash mismatch")
            if item["role"] == "ORIGINAL":
                if any(key in item for key in ("parent_variant_id", "mutation_rule_id", "mutation_family", "patch")):
                    _fail(f"{item['variant_id']} original has mutation metadata")
            else:
                if item.get("parent_variant_id") != original_id:
                    _fail(f"{item['variant_id']} does not point to original variant")
                required_mutation_fields = {
                    "mutation_rule_id", "mutation_family", "operation", "anchor_sha256",
                    "replacement_sha256", "occurrence", "patch",
                }
                if not required_mutation_fields <= set(item):
                    _fail(f"{item['variant_id']} lacks mutation provenance")
                rule = rule_map.get(item.get("mutation_rule_id"))
                if rule is None or rule["family"] != item.get("mutation_family"):
                    _fail(f"{item['variant_id']} mutation rule mismatch")
                patch_path = _relative_file(repository_root, item["patch"]["path"], label=f"{item['variant_id']} patch")
                if sha256_file(patch_path) != item["patch"]["sha256"]:
                    _fail(f"{item['variant_id']} patch hash mismatch")
        original_workloads = [item for item in kernel_workloads if item["role"] == "ORIGINAL"]
        boundary_workloads = [item for item in kernel_workloads if item["role"] == "BOUNDARY"]
        if len(original_workloads) != lock["policy"]["original_workloads"]:
            _fail(f"{kernel_id} must have exactly one original workload")
        if len(boundary_workloads) != lock["policy"]["boundary_workloads"]:
            _fail(f"{kernel_id} must have exactly three boundary workloads")
        original = original_workloads[0]
        dimensions: set[str] = set()
        for workload in kernel_workloads:
            recipe_path = _relative_file(repository_root, workload["recipe"]["path"], label=workload["spec_id"])
            if sha256_file(recipe_path) != workload["recipe"]["sha256"]:
                _fail(f"{workload['spec_id']} recipe hash mismatch")
            recipe = load_json(recipe_path)
            _schema_validate(recipe, schema_root / "mutation-workload-v1.schema.json")
            if recipe["schema_version"] != WORKLOAD_VERSION or recipe["bindings"] != workload["bindings"]:
                _fail(f"{workload['spec_id']} recipe identity mismatch")
            if workload["role"] == "ORIGINAL":
                if workload.get("parent_spec_id") is not None or workload["delta"] or workload["boundary_dimensions"]:
                    _fail(f"{workload['spec_id']} original workload has boundary metadata")
            else:
                if workload.get("parent_spec_id") != original["spec_id"]:
                    _fail(f"{workload['spec_id']} does not point to original workload")
                actual = _diff_bindings(original["bindings"], workload["bindings"])
                if actual != workload["delta"]:
                    _fail(f"{workload['spec_id']} declared workload delta mismatch")
                actual_dimensions = {item["path"].split("/")[2] for item in actual}
                if set(workload["boundary_dimensions"]) != actual_dimensions:
                    _fail(f"{workload['spec_id']} boundary dimensions mismatch")
                dimensions.update(actual_dimensions)
        if len(dimensions) < lock["policy"]["minimum_boundary_dimensions"]:
            _fail(f"{kernel_id} lacks independent workload boundary dimensions")

        expected_pairs = {
            (variant["variant_id"], workload["spec_id"])
            for variant in kernel_variants
            for workload in kernel_workloads
        }
        actual_pairs = {
            (case["variant_id"], case["spec_id"])
            for case in cases.values()
            if case["kernel_id"] == kernel_id
        }
        if actual_pairs != expected_pairs:
            _fail(f"{kernel_id} execution matrix is incomplete")

    for case in cases.values():
        variant = variants.get(case["variant_id"])
        workload = workloads.get(case["spec_id"])
        if variant is None or workload is None:
            _fail(f"{case['case_id']} references unknown variant or workload")
        if variant["kernel_id"] != case["kernel_id"] or workload["kernel_id"] != case["kernel_id"]:
            _fail(f"{case['case_id']} combines records from different kernels")

    expected_counts = {
        "kernels": len(kernels),
        "original_variants": sum(v["role"] == "ORIGINAL" for v in variants.values()),
        "mutant_variants": sum(v["role"] == "MUTANT" for v in variants.values()),
        "original_workloads": sum(w["role"] == "ORIGINAL" for w in workloads.values()),
        "boundary_workloads": sum(w["role"] == "BOUNDARY" for w in workloads.values()),
        "cases": len(cases),
        "assessed_cases": len(baseline["results"]),
    }
    if any(baseline["counts"].get(key) != value for key, value in expected_counts.items()):
        _fail(f"baseline structural counts differ: expected {expected_counts}, got {baseline['counts']}")
    runtime_counts = {
        "lowered_cases": sum(item["lowering_status"] == "LOWERED" for item in baseline["results"]),
        "captured_snapshot_cases": sum(item["capture_status"] == "CAPTURED" for item in baseline["results"]),
        "parsed_pgir_cases": sum(item["pgir_status"] == "PARSED" for item in baseline["results"]),
        "checked_cases": sum(item["checker_status"] == "CHECKED" for item in baseline["results"]),
        "numeric_pass_cases": sum(item["numeric_status"] == "PASS" for item in baseline["results"]),
        "numeric_mismatch_cases": sum(item["numeric_status"] == "NUMERIC_MISMATCH" for item in baseline["results"]),
        "numeric_hang_cases": sum(item["numeric_status"] == "HANG" for item in baseline["results"]),
        "numeric_compile_error_cases": sum(item["numeric_status"] == "COMPILE_ERROR" for item in baseline["results"]),
    }
    for key, value in runtime_counts.items():
        if key in baseline["counts"] and baseline["counts"][key] != value:
            _fail(f"baseline runtime count {key} differs: expected {value}, got {baseline['counts'][key]}")
    result_ids = [item["case_id"] for item in baseline["results"]]
    if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= set(cases):
        _fail("baseline result IDs are duplicate or unknown")

    summary = MutationSuiteSummary(
        kernels=expected_counts["kernels"],
        original_variants=expected_counts["original_variants"],
        mutant_variants=expected_counts["mutant_variants"],
        original_workloads=expected_counts["original_workloads"],
        boundary_workloads=expected_counts["boundary_workloads"],
        cases=expected_counts["cases"],
    )
    return lock, catalog, baseline, summary
