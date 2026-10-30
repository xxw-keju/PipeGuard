"""Final-device-code checks that cannot be justified from pre-lowering PGIR.

The register check deliberately supports straight-line WGMMA issue prefixes.
It does not infer control flow from textual instruction order, or treat a
shared-memory proxy fence / wait_group as a register fence.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _operands(text: str) -> list[str]:
    return re.split(r",(?![^{}]*\})", text)


def check_wgmma_register_fences(text: str) -> dict[str, Any]:
    # Preserve line numbers, including multi-line block comments.
    clean = re.sub(r"/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)
    clean = re.sub(r"//[^\n]*", "", clean)
    instructions: list[tuple[int, str, str, str]] = []
    unparsed: list[int] = []
    for line, raw in enumerate(clean.splitlines(), 1):
        raw = re.sub(r"^\s*[\w$]+:\s*", "", raw).strip()
        if ((raw.startswith(("{", "}")) and raw not in {"{", "}"})
            or (".entry" in raw and "{" in raw and ";" in raw)):
            unparsed.append(line)
        if not raw or raw.startswith((".", "{", "}", ")")):
            continue
        if not raw.endswith(";"):
            unparsed.append(line)
        for statement in raw.split(";"):
            match = re.fullmatch(r"\s*(?:@(!?%[\w$]+)\s+)?([\w.:]+)(?:\s+(.*))?\s*", statement)
            if match:
                instructions.append((line, match[1] or "", match[2], match[3] or ""))
            elif statement.strip():
                unparsed.append(line)
    mma_indices = [i for i, (_, _, op, _) in enumerate(instructions) if op.startswith("wgmma.mma_async.")]
    result: dict[str, Any] = {
        "check": "wgmma_register_fences", "status": "CHECKED",
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "mma_instructions": len(mma_indices), "diagnostics": [],
    }
    textual_mmas = len(re.findall(r"\bwgmma\.mma_async\.", clean))
    if not textual_mmas:
        result["status"] = "NOT_APPLICABLE"
        return result

    def unsupported(line: int, reason: str) -> dict[str, Any]:
        result["status"] = "UNSUPPORTED"
        # A textual prefix is not a witness if its reachability is unknown.
        result["diagnostics"] = [{"code": "PTX_WGMMA_REGISTER_ORDER_UNRESOLVED",
                                  "message": reason, "details": {"line": line}}]
        return result

    if unparsed or len(mma_indices) != textual_mmas:
        return unsupported(unparsed[0] if unparsed else 0,
                           "PTX instruction formatting requires a complete parser")
    if len(re.findall(r"\.entry\b", clean)) != 1 or re.search(r"\.func\b", clean):
        return unsupported(0, "register ordering requires one kernel without device functions")
    # A backedge after the last textual MMA can execute that MMA again with
    # modified registers. Do not certify just the first iteration's prefix.
    for line, _, opcode, _ in instructions:
        if opcode.split(".")[0] in {"bra", "brx", "call"}:
            return unsupported(line, "control flow requires path-sensitive register analysis")
    fence_line = 0
    ordinary: dict[str, tuple[int, bool]] = {}
    last_mma: dict[str, tuple[int, str, str]] = {}
    predicates: dict[str, bool] = {}
    for line, pred, opcode, args in instructions[:mma_indices[-1] + 1]:
        enabled = predicates.get(pred.lstrip("!")) if pred else True
        if pred.startswith("!") and enabled is not None:
            enabled = not enabled
        if enabled is False:
            continue
        base = opcode.split(".")[0]
        if base in {"bra", "brx", "call", "ret", "exit", "trap"}:
            return unsupported(line, "control flow before WGMMA requires path-sensitive register analysis")
        if opcode == "wgmma.fence.sync.aligned":
            if enabled is None:
                return unsupported(line, "predicate of collective register fence is unresolved")
            fence_line = line
            ordinary.clear()
            last_mma.clear()
            continue
        if opcode.startswith("wgmma.mma_async."):
            if enabled is None:
                return unsupported(line, "predicate of collective WGMMA issue is unresolved")
            operands = [x.strip() for x in _operands(args)]
            if len(operands) < 4 or not operands[0].startswith("{"):
                return unsupported(line, "WGMMA operand register lists are malformed or unavailable")
            accum = re.findall(r"%[\w$]+", operands[0])
            input_a = re.findall(r"%[\w$]+", operands[1]) if operands[1].startswith("{") else []
            for operand in [operands[0], *([operands[1]] if input_a else [])]:
                if any(re.fullmatch(r"%[\w$]+", reg.strip()) is None
                       for reg in operand.strip("{}").split(",")):
                    return unsupported(line, "WGMMA register naming is unsupported")
            shape_match = re.search(r"\.m\d+n\d+k\d+\.[\w.]+", opcode)
            if not accum or shape_match is None:
                return unsupported(line, "WGMMA shape or accumulator registers are unavailable")
            shape = shape_match[0]
            missing: dict[str, int] = {}
            for reg in accum + input_a:
                if reg in ordinary:
                    access_line, definite = ordinary[reg]
                    if not definite:
                        return unsupported(line, "predicated ordinary register access requires path analysis")
                    missing[reg] = access_line
                if reg in last_mma:
                    previous_line, previous_shape, role = last_mma[reg]
                    accumulator_chain = reg in accum and role == "accumulator" and previous_shape == shape
                    input_reads = reg in input_a and role == "input_a"
                    if not (accumulator_chain or input_reads):
                        missing[reg] = previous_line
            if not fence_line or missing:
                result["diagnostics"].append({
                    "code": "WGMMA_REGISTER_FENCE_MISSING",
                    "message": "WGMMA requires a register fence after prior ordinary register accesses",
                    "details": {"mma_line": line, "last_fence_line": fence_line or None,
                                "prior_accesses": missing, "instruction": opcode + " " + args},
                })
            for reg in accum:
                last_mma[reg] = (line, shape, "accumulator")
            for reg in input_a:
                last_mma[reg] = (line, shape, "input_a")
            continue
        if opcode.startswith("wgmma."):
            if opcode not in {"wgmma.commit_group.sync.aligned", "wgmma.wait_group.sync.aligned"}:
                return unsupported(line, "unrecognized WGMMA operation")
            continue
        # Both ordinary reads and writes need ordering before a later MMA.
        for reg in re.findall(r"%[\w$]+", args):
            previous = ordinary.get(reg)
            if enabled is True or previous is None:
                ordinary[reg] = (line, enabled is True)
        operands = [x.strip() for x in _operands(args)]
        if operands and operands[0].startswith("%"):
            predicates.pop(operands[0], None)
            if opcode == "mov.pred" and enabled is True and len(operands) == 2:
                if operands[1] in {"0", "1", "-1"}:
                    predicates[operands[0]] = operands[1] != "0"
    return result


def apply_device_checks(
    result: dict[str, Any], artifact: dict[str, str] | None, *, enabled: bool = True,
) -> dict[str, Any]:
    if not enabled or artifact is None:
        result = dict(result)
        if "manifest" in result:
            manifest = dict(result["manifest"])
            provenance = dict(manifest.get("frontend_provenance", {}))
            if "final_ptx" in provenance:
                provenance["final_ptx"] = {"sha256": provenance["final_ptx"].get("sha256")}
                manifest["frontend_provenance"] = provenance
                result["manifest"] = manifest
        return {**result, "validation_scope": "PGIR",
                "device_validation": {"wgmma_register_fences": "DISABLED" if not enabled else "NOT_PROVIDED"}}
    text = artifact.get("text") if isinstance(artifact, dict) else None
    digest = artifact.get("sha256") if isinstance(artifact, dict) else None
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != digest:
        review = {"status": "UNSUPPORTED", "diagnostics": [{
            "code": "FINAL_PTX_EVIDENCE_INVALID", "message": "final PTX text does not match its SHA256",
        }]}
    else:
        review = check_wgmma_register_fences(text)
    result = dict(result)
    result["validation_scope"] = "PGIR_AND_FINAL_PTX_WGMMA_REGISTERS"
    result["device_validation"] = {"wgmma_register_fences": review["status"]}
    result["artifacts"] = {**result.get("artifacts", {}), "final_ptx": review}
    findings = review["diagnostics"]
    if findings:
        result["diagnostics"] = [*result.get("diagnostics", []), *findings]
        if result["verdict"] == "SAFE":
            result["verdict"] = "UNSUPPORTED" if review["status"] == "UNSUPPORTED" else "SYNC_HAZARD"
            result["summary"] = findings[0]["message"]
    # Keep the evidence in serialized PGIR, but avoid duplicating kernel source
    # in each checker report's manifest.
    if "manifest" in result:
        manifest = dict(result["manifest"])
        provenance = dict(manifest.get("frontend_provenance", {}))
        if "final_ptx" in provenance:
            provenance["final_ptx"] = {"sha256": digest}
            manifest["frontend_provenance"] = provenance
            result["manifest"] = manifest
    return result
