"""Suite 2 admission gate 2: label grounding in the lowered CUDA/TTGIR artifact.

For every case in a suite2 full-recapture run directory, checks that the
construct named by the case's (primitive, result_class) label demonstrably
survived compilation, by matching REQUIRED and FORBIDDEN regex patterns over
the lowered artifact:

  * tilelang: ``<run-dir>/cases/<case_id>/kernel.cu`` (final CUDA source)
  * triton:   ``<run-dir>/cases/<case_id>/raw.ttgir.mlir`` (TTGIR)

Statuses:
  * PASS         — every required pattern matched, no forbidden pattern hit.
  * NEEDS_REVIEW — a non-strict required pattern missed, a non-strict
                   forbidden pattern hit, the artifact is missing, or no
                   rule covers the family. Conservative by design: text-level
                   evidence is advisory unless the config marks it strict.
  * FAIL         — a strict required pattern missed or a strict forbidden
                   pattern hit.

The built-in default table was derived by inspecting the lowered artifacts
of the suite2-210 freeze and updated for the 2026-10 structural-diversity
redesign (suite/suite2/REDESIGN-TILELANG.md, REDESIGN-TRITON.md): with five
deliberately distinct protocol skeletons per cell, each family rule is the
*intersection* of what all five variants must exhibit (see
suite/suite2/ADMISSION-GATES.md for what this text-level gate can and cannot
establish; the authoritative grounding is the capture snapshot plus execution
evidence). Override or extend it with
``--config rules.json`` (same shape as ``--dump-config`` output; a family
key present in the override replaces the built-in rule for that family).

Exit status: 0 if no case FAILs (NEEDS_REVIEW allowed; use --fail-on-review
to also exit nonzero on NEEDS_REVIEW), 1 otherwise, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pipeguard-suite2-cuda-grounding-v1"
MAX_EVIDENCE_LINES = 20

ARTIFACTS = {"tilelang": "kernel.cu", "triton": "raw.ttgir.mlir"}


def rule(required: list[tuple[str, str, bool]], forbidden: list[tuple[str, str, bool]] = ()) -> dict[str, Any]:
    return {
        "required": [{"name": name, "pattern": pattern, "strict": strict} for name, pattern, strict in required],
        "forbidden": [{"name": name, "pattern": pattern, "strict": strict} for name, pattern, strict in forbidden],
    }


# Family table. Keys: "<dsl>:<primitive>:<result_class>", with "<dsl>:<primitive>:*"
# as fallback. Patterns were derived from the suite2-210 freeze lowered
# artifacts and reworked for the structural-diversity redesign
# (REDESIGN-TILELANG.md / REDESIGN-TRITON.md): each rule is the intersection
# of all five per-cell skeletons, verified against the redesigned recapture.
DEFAULT_RULES: dict[str, dict[str, Any]] = {
    # ---------------- tilelang (kernel.cu) ----------------
    # The labeled construct for Barrier is __syncthreads itself. SAFE must
    # have it; SYNC_HAZARD keeps it (the hazard is divergent/conditional
    # execution, which text matching cannot establish). DATA_HAZARD variants
    # may legitimately contain a *non-repairing* barrier (too late, wrong
    # scope, or the mbarrier-init barrier), so no forbid applies; position is
    # not checkable at text level.
    "tilelang:Barrier:SAFE": rule([("syncthreads", r"__syncthreads\s*\(", True)]),
    "tilelang:Barrier:SYNC_HAZARD": rule([("syncthreads", r"__syncthreads\s*\(", True)]),
    "tilelang:Barrier:DATA_HAZARD": rule([("kernel_entry", r"__global__\s+void", True)]),
    # Memory: plain global/shared accesses. SYNC_HAZARD's labeled construct
    # is *some* synchronization that fails (CTA barrier, named partial
    # barrier, or mbarrier wait). DATA_HAZARD may carry a non-repairing
    # barrier (see Barrier above), so no forbid.
    "tilelang:Memory:SAFE": rule([("kernel_entry", r"__global__\s+void", True)]),
    "tilelang:Memory:DATA_HAZARD": rule([("kernel_entry", r"__global__\s+void", True)]),
    "tilelang:Memory:SYNC_HAZARD": rule([
        ("sync_construct", r"__syncthreads\s*\(|__sync_thread_partial\(|\.wait\(", True),
    ]),
    # mbarrier family: mbarrier intrinsics must survive in kernel.cu.
    # DATA_HAZARD variants include missing-wait and dead-guarded protocols,
    # so only the barrier allocation (init) is common to all five.
    "tilelang:mbarrier:DATA_HAZARD": rule([("mbarrier_init", r"\.init\(", True)]),
    "tilelang:mbarrier:*": rule([
        ("mbarrier_init", r"\.init\(", True),
        ("mbarrier_arrive", r"\.arrive\(\)", True),
        ("mbarrier_wait", r"\.wait\(", True),
        ("fence_barrier_init", r"tl::fence_barrier_init", True),
    ]),
    # TMA: the suite has both load-side (mbarrier-completed) and store-side
    # (tma_store_wait-completed) variants; DATA_HAZARD includes wait-removed
    # idioms, so only the TMA op itself is common to the hazard cell.
    "tilelang:TMA:SAFE": rule([
        ("tma_op", r"tl::tma_(load|store)\(", True),
        ("completion_wait", r"\.wait\(|tl::tma_store_wait", True),
    ]),
    "tilelang:TMA:DATA_HAZARD": rule([("tma_op", r"tl::tma_(load|store)\(", True)]),
    "tilelang:TMA:SYNC_HAZARD": rule([
        ("tma_load", r"tl::tma_load\(", True),
        ("mbarrier_init", r"\.init\(", True),
        ("mbarrier_wait", r"\.wait\(", True),
    ]),
    # cp.async: DATA_HAZARD includes commit-without-wait idioms (no wait);
    # SYNC_HAZARD includes wait-before-commit / divergent-commit idioms
    # (no unconditional commit).
    "tilelang:cp.async:SAFE": rule([
        ("cp_async_issue", r"tl::cp_async_gs<", True),
        ("cp_async_commit", r"tl::cp_async_commit\(", True),
        ("cp_async_wait", r"tl::cp_async_wait<", True),
    ]),
    "tilelang:cp.async:DATA_HAZARD": rule([
        ("cp_async_issue", r"tl::cp_async_gs", True),
        ("cp_async_commit", r"tl::cp_async_commit|cp\.async\.commit_group", True),
    ]),
    "tilelang:cp.async:SYNC_HAZARD": rule([
        ("cp_async_issue", r"tl::cp_async_gs", True),
        ("cp_async_wait", r"tl::cp_async_wait|cp\.async\.wait_group", True),
    ]),
    "tilelang:WGMMA:*": rule([
        ("wgmma_issue", r"tl::wgmma_", True),
        ("warpgroup_fence_operand", r"tl::warpgroup_fence_operand", True),
    ]),
    # Proxy: SAFE needs the async-proxy fence plus a TMA op (load- and
    # store-side variants both exist). DATA_HAZARD variants include
    # second-generation-unfenced idioms where an earlier fence legitimately
    # appears, so presence of a fence is not forbidden; the hazard is the
    # *placement*, which text matching cannot establish. SYNC_HAZARD lowers
    # through the wgmma pipeline in this suite.
    "tilelang:Proxy:SAFE": rule([
        ("fence_proxy_async", r"tl::fence_proxy_async", True),
        ("tma_op", r"tl::tma_(load|store)\(", True),
    ]),
    "tilelang:Proxy:DATA_HAZARD": rule([("tma_op", r"tl::tma_(load|store)\(", True)]),
    "tilelang:Proxy:SYNC_HAZARD": rule([
        ("fence_proxy_async", r"tl::fence_proxy_async", True),
        ("wgmma_issue", r"tl::wgmma_", True),
    ]),
    # ---------------- triton (raw.ttgir.mlir) ----------------
    "triton:Barrier:SAFE": rule([("ttg_barrier", r"\bttg\.barrier\b", True)]),
    "triton:Barrier:SYNC_HAZARD": rule([
        ("inline_asm_barrier", r"elementwise_inline_asm", True),
        ("divergent_bar_sync", r"bar\.sync", True),
    ]),
    "triton:Barrier:DATA_HAZARD": rule(
        [("store_op", r"\bttg\.local_store\b|\btt\.store\b", True)],
        [("ttg_barrier", r"\bttg\.barrier\b", False)],
    ),
    # Memory: the write-write variants (disjoint/same-address WAW) carry no
    # tt.load at all, so only the store is common to all five skeletons.
    "triton:Memory:SAFE": rule([("store", r"\btt\.store\b", True)]),
    "triton:Memory:DATA_HAZARD": rule(
        [("store", r"\btt\.store\b", True)],
        [("ttg_barrier", r"\bttg\.barrier\b", False)],
    ),
    "triton:Memory:SYNC_HAZARD": rule([
        ("divergent_bar_sync", r"bar\.sync", True),
        ("store", r"\btt\.store\b", True),
    ]),
    "triton:mbarrier:SAFE": rule([
        ("init_barrier", r"\bttng\.init_barrier\b", True),
        ("arrive_barrier", r"\bttng\.arrive_barrier\b", True),
        ("wait_barrier", r"\bttng\.wait_barrier\b", True),
    ]),
    "triton:mbarrier:SYNC_HAZARD": rule([
        ("init_barrier", r"\bttng\.init_barrier\b", True),
        ("arrive_barrier", r"\bttng\.arrive_barrier\b", True),
        ("wait_barrier", r"\bttng\.wait_barrier\b", True),
    ]),
    # The redesigned mbarrier DATA_HAZARD family uses plain arrive/wait
    # protocols (no TMA transaction); the missing-wait variant legitimately
    # carries no wait_barrier, so only init + arrive are common to all five.
    "triton:mbarrier:DATA_HAZARD": rule([
        ("init_barrier", r"\bttng\.init_barrier\b", True),
        ("arrive_barrier", r"\bttng\.arrive_barrier\b", True),
    ]),
    # TMA DATA_HAZARD includes missing-wait idioms, so wait_barrier is not
    # required there; SAFE and SYNC_HAZARD always wait.
    "triton:TMA:DATA_HAZARD": rule([
        ("tma_copy_g2l", r"\bttng\.async_tma_copy_global_to_local\b", True),
        ("barrier_expect", r"\bttng\.barrier_expect\b", True),
        ("init_barrier", r"\bttng\.init_barrier\b", True),
    ]),
    "triton:TMA:*": rule([
        ("tma_copy_g2l", r"\bttng\.async_tma_copy_global_to_local\b", True),
        ("barrier_expect", r"\bttng\.barrier_expect\b", True),
        ("init_barrier", r"\bttng\.init_barrier\b", True),
        ("wait_barrier", r"\bttng\.wait_barrier\b", True),
    ]),
    # cp.async DATA_HAZARD includes missing-wait idioms (no async_wait);
    # SYNC_HAZARD includes wait-before-commit / bare-wait idioms (no
    # async_commit_group); only SAFE requires the full issue/commit/wait set.
    "triton:cp.async:SAFE": rule([
        ("async_copy", r"\bttg\.async_copy_global_to_local\b", True),
        ("async_commit", r"\bttg\.async_commit_group\b", True),
        ("async_wait", r"\bttg\.async_wait\b", True),
    ]),
    "triton:cp.async:DATA_HAZARD": rule([
        ("async_copy", r"\bttg\.async_copy_global_to_local\b", True),
        ("async_commit", r"\bttg\.async_commit_group\b", True),
    ]),
    "triton:cp.async:SYNC_HAZARD": rule([
        ("async_copy", r"\bttg\.async_copy_global_to_local\b", True),
        ("async_wait", r"\bttg\.async_wait\b", True),
    ]),
    "triton:WGMMA:*": rule([
        ("warp_group_dot", r"\bttng\.warp_group_dot\b", True),
        ("warp_group_dot_wait", r"\bttng\.warp_group_dot_wait\b", True),
    ]),
    "triton:Proxy:SAFE": rule([
        ("fence_async_shared", r"\bttng\.fence_async_shared\b", True),
        ("tma_copy_l2g", r"\bttng\.async_tma_copy_local_to_global\b", True),
        ("tma_store_wait", r"\bttng\.async_tma_store_wait\b", True),
    ]),
    "triton:Proxy:DATA_HAZARD": rule(
        [
            ("tma_copy_l2g", r"\bttng\.async_tma_copy_local_to_global\b", True),
            ("tma_store_wait", r"\bttng\.async_tma_store_wait\b", True),
        ],
        [("fence_async_shared", r"\bttng\.fence_async_shared\b", False)],
    ),
    # The Proxy SYNC_HAZARD family lowers through the wgmma pipeline; the
    # proxy-fence placement hazard itself is not visible at text level.
    "triton:Proxy:SYNC_HAZARD": rule([
        ("warp_group_dot", r"\bttng\.warp_group_dot\b", True),
        ("warp_group_dot_wait", r"\bttng\.warp_group_dot_wait\b", True),
    ]),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION + "-config",
    "artifacts": ARTIFACTS,
    "rules": DEFAULT_RULES,
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = {
        "schema_version": DEFAULT_CONFIG["schema_version"],
        "artifacts": dict(DEFAULT_CONFIG["artifacts"]),
        "rules": dict(DEFAULT_CONFIG["rules"]),
    }
    if path is not None:
        override = json.loads(path.read_text())
        config["artifacts"].update(override.get("artifacts", {}))
        config["rules"].update(override.get("rules", {}))
    return config


def find_rule(rules: dict[str, Any], dsl: str, primitive: str, result_class: str) -> tuple[str, dict[str, Any]] | None:
    for key in (f"{dsl}:{primitive}:{result_class}", f"{dsl}:{primitive}:*"):
        if key in rules:
            return key, rules[key]
    return None


def artifact_lines(path: Path, dsl: str) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    if dsl == "triton":
        # MLIR #loc definitions embed source file paths (which contain family
        # names like ".../barrier/..."); blank them so patterns only see IR.
        lines = ["" if line.lstrip().startswith("#loc") else line for line in lines]
    return lines


def match_pattern(pattern: str, lines: list[str]) -> list[dict[str, Any]]:
    compiled = re.compile(pattern)
    hits = []
    for number, line in enumerate(lines, 1):
        if compiled.search(line):
            hits.append({"line": number, "text": line.strip()[:160]})
            if len(hits) >= MAX_EVIDENCE_LINES:
                break
    return hits


def check_case(case_dir: Path, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    dsl = str(spec.get("dsl", "UNKNOWN"))
    primitive = str(spec.get("primitive", "UNKNOWN"))
    result_class = str(spec.get("result_class", "UNKNOWN"))
    record: dict[str, Any] = {
        "case_id": spec.get("case_id", case_dir.name),
        "dsl": dsl,
        "primitive": primitive,
        "result_class": result_class,
        "artifact": None,
        "rule_key": None,
        "required": [],
        "forbidden": [],
        "reasons": [],
    }
    artifact_name = config["artifacts"].get(dsl)
    if artifact_name is None:
        record["status"] = "NEEDS_REVIEW"
        record["reasons"].append(f"NO_ARTIFACT_MAPPING_FOR_DSL:{dsl}")
        return record
    artifact_path = case_dir / artifact_name
    record["artifact"] = artifact_name
    located = find_rule(config["rules"], dsl, primitive, result_class)
    if located is None:
        record["status"] = "NEEDS_REVIEW"
        record["reasons"].append("NO_RULE_FOR_FAMILY")
        return record
    record["rule_key"] = located[0]
    if not artifact_path.is_file():
        record["status"] = "NEEDS_REVIEW"
        record["reasons"].append("MISSING_ARTIFACT")
        return record
    lines = artifact_lines(artifact_path, dsl)
    status = "PASS"

    def degrade(current: str, strict: bool) -> str:
        if strict:
            return "FAIL"
        return current if current == "FAIL" else "NEEDS_REVIEW"

    for entry in located[1].get("required", []):
        hits = match_pattern(entry["pattern"], lines)
        matched = bool(hits)
        record["required"].append({
            "name": entry["name"],
            "pattern": entry["pattern"],
            "strict": bool(entry.get("strict", False)),
            "matched": matched,
            "evidence": hits,
        })
        if not matched:
            status = degrade(status, bool(entry.get("strict", False)))
            record["reasons"].append(f"REQUIRED_MISS:{entry['name']}")
    for entry in located[1].get("forbidden", []):
        hits = match_pattern(entry["pattern"], lines)
        record["forbidden"].append({
            "name": entry["name"],
            "pattern": entry["pattern"],
            "strict": bool(entry.get("strict", False)),
            "hit": bool(hits),
            "evidence": hits,
        })
        if hits:
            status = degrade(status, bool(entry.get("strict", False)))
            record["reasons"].append(f"FORBIDDEN_HIT:{entry['name']}")
    record["status"] = status
    return record


def run(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    cases_dir = run_dir / "cases"
    if not cases_dir.is_dir():
        raise RuntimeError(f"not a suite2 run directory (missing {cases_dir})")
    records = []
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        spec_path = case_dir / "normalized-spec.json"
        if not spec_path.is_file():
            records.append({
                "case_id": case_dir.name,
                "status": "NEEDS_REVIEW",
                "reasons": ["MISSING_NORMALIZED_SPEC"],
                "dsl": None, "primitive": None, "result_class": None,
                "artifact": None, "rule_key": None, "required": [], "forbidden": [],
            })
            continue
        records.append(check_case(case_dir, json.loads(spec_path.read_text()), config))
    return records


def render_table(records: list[dict[str, Any]]) -> str:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[":".join(str(record.get(k)) for k in ("dsl", "primitive", "result_class"))].append(record)
    header = f"{'family (dsl:primitive:result_class)':<42} {'n':>3} {'pass':>5} {'review':>6} {'fail':>5}"
    lines = [header, "-" * len(header)]
    for key, members in sorted(families.items()):
        counts = Counter(member["status"] for member in members)
        lines.append(
            f"{key:<42} {len(members):>3} {counts.get('PASS', 0):>5} "
            f"{counts.get('NEEDS_REVIEW', 0):>6} {counts.get('FAIL', 0):>5}"
        )
    flagged = [record for record in records if record["status"] != "PASS"]
    if flagged:
        lines.append("")
        lines.append("non-PASS cases:")
        for record in sorted(flagged, key=lambda item: item["case_id"]):
            lines.append(f"  {record['case_id']:<44} {record['status']:<13} {';'.join(record['reasons'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="suite2 full-recapture output directory")
    parser.add_argument("--config", type=Path, default=None, help="JSON rules override (same shape as --dump-config)")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here (default: no file)")
    parser.add_argument("--dump-config", action="store_true", help="print the effective config as JSON and exit")
    parser.add_argument("--fail-on-review", action="store_true", help="exit nonzero on NEEDS_REVIEW as well as FAIL")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dump_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --dump-config is given")

    run_dir = args.run_dir.resolve()
    records = run(run_dir, config)
    statuses = Counter(record["status"] for record in records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "config_override": str(args.config) if args.config else None,
        "cases": len(records),
        "statuses": dict(statuses),
        "records": sorted(records, key=lambda item: item["case_id"]),
        "note": (
            "Text-level grounding is advisory: it establishes that the labeled "
            "construct survived lowering, not that the semantic hazard/safety "
            "property holds. Authoritative grounding is the capture snapshot "
            "plus execution evidence (see suite/suite2/ADMISSION-GATES.md)."
        ),
    }
    print(render_table(records))
    print()
    print(json.dumps({"cases": len(records), "statuses": dict(sorted(statuses.items()))}, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {args.out}")
    if statuses.get("FAIL"):
        return 1
    if args.fail_on_review and statuses.get("NEEDS_REVIEW"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
