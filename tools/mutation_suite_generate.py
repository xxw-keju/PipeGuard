from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutation_suite_triton_faults import RULES as TRITON_FAULT_RULES
from mutation_suite_triton_faults import STRATEGIES as TRITON_FAULT_STRATEGIES
from mutation_suite_triton_faults import mutate_source as inject_triton_fault


SUITE_ID = "pipeguard-matched-official-mutations-sm90-v2"
RULES = [
    {
        "rule_id": "MUT-M1-PIPE-STAGE-LOW",
        "family": "PIPELINE_DEPTH",
        "name": "M1 low pipeline depth",
        "description": "Set the selected software-pipeline loop to one stage.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M2-PIPE-STAGE-ALT",
        "family": "PIPELINE_DEPTH",
        "name": "M2 alternate pipeline depth",
        "description": "Set the selected software-pipeline loop to a second non-original depth.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M3-DEST-INIT-DUP",
        "family": "BUFFER_INIT",
        "name": "M3 duplicate destination initialization",
        "description": "Repeat or add an overwritten destination/accumulator initialization.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M4-SOURCE-A-INIT",
        "family": "BUFFER_INIT",
        "name": "M4 initialize first producer buffer",
        "description": "Initialize the first producer destination immediately before it is overwritten.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M5-SOURCE-B-INIT",
        "family": "BUFFER_INIT",
        "name": "M5 initialize second producer buffer",
        "description": "Initialize the second producer destination immediately before it is overwritten.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M6-PRODUCER-DUP-BEFORE",
        "family": "ASYNC_PRODUCER",
        "name": "M6 duplicate producer before compute",
        "description": "Repeat a selected source load or copy immediately before its consumer.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
    {
        "rule_id": "MUT-M7-PRODUCER-DUP-AFTER",
        "family": "ASYNC_PRODUCER",
        "name": "M7 duplicate producer after compute",
        "description": "Repeat a selected source load or copy immediately after its consumer.",
        "computational_intent": "PRESERVE_OUTPUT",
    },
] + list(TRITON_FAULT_RULES)
RULE_BY_ID = {item["rule_id"]: item for item in RULES}


@dataclass(frozen=True)
class Mutation:
    suffix: str
    rule_id: str
    operation: str
    marker: str
    value: str | None = None
    occurrence: int = 1
    target: str | None = None
    target_occurrence: int = 1


@dataclass(frozen=True)
class KernelDefinition:
    kernel_id: str
    operator_family: str
    project: str
    dsl: str
    source_id: str
    candidate_id: str
    features: tuple[str, ...]
    mutations: tuple[Mutation, ...]


M1 = "MUT-M1-PIPE-STAGE-LOW"
M2 = "MUT-M2-PIPE-STAGE-ALT"
M3 = "MUT-M3-DEST-INIT-DUP"
M4 = "MUT-M4-SOURCE-A-INIT"
M5 = "MUT-M5-SOURCE-B-INIT"
M6 = "MUT-M6-PRODUCER-DUP-BEFORE"
M7 = "MUT-M7-PRODUCER-DUP-AFTER"

def triton_fault_mutations(kernel_id: str) -> tuple[Mutation, ...]:
    return tuple(
        Mutation(
            suffix=strategy.suffix,
            rule_id=strategy.rule_id,
            operation="inject_canary",
            marker=strategy.suffix,
            value=strategy.rule_id,
        )
        for strategy in TRITON_FAULT_STRATEGIES.values()
        if strategy.kernel_ids is None or kernel_id in strategy.kernel_ids
    )


KERNELS = (
    KernelDefinition(
        "triton-dense-gemm", "DENSE_GEMM", "Triton", "triton", "triton-61c02bcd", "965d9fbea6db833b",
        ("PIPELINED", "CP_ASYNC", "MMA"),
        (
            Mutation("m1-stage1", M1, "replace", "for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):", "for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=1):"),
            Mutation("m2-stage2", M2, "replace", "for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):", "for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=2):"),
            Mutation("m3-output-init", M3, "insert_before", "tl.store(c_ptrs, c, mask=c_mask)", "tl.store(c_ptrs, 0.0, mask=c_mask)"),
            Mutation("m4-output-dup", M4, "duplicate", "tl.store(c_ptrs, c, mask=c_mask)"),
        ) + triton_fault_mutations("triton-dense-gemm"),
    ),
    KernelDefinition(
        "tilelang-dense-gemm", "DENSE_GEMM", "TileLang", "tilelang", "tilelang-2c31e071", "4d6009bd3a002eaa",
        ("PIPELINED", "TMA", "MMA"),
        (
            Mutation("m1-stage1", M1, "replace", "num_stages=3", "num_stages=1"),
            Mutation("m2-stage2", M2, "replace", "num_stages=3", "num_stages=2"),
            Mutation("m3-acc-init", M3, "duplicate", "T.clear(C_local)"),
            Mutation("m4-a-init", M4, "insert_before", "T.copy(A[", "T.clear(A_shared)"),
            Mutation("m5-b-init", M5, "insert_before", "T.copy(B[", "T.clear(B_shared)"),
            Mutation("m6-a-before", M6, "duplicate", "T.copy(A["),
            Mutation("m7-a-after", M7, "duplicate_after", "T.copy(A[", target="T.gemm(A_shared"),
        ),
    ),
    KernelDefinition(
        "triton-grouped-gemm", "GROUPED_GEMM", "Triton", "triton", "triton-61c02bcd", "f0f61ca3e2f347dc",
        ("PIPELINED", "PERSISTENT", "MMA", "IRREGULAR_GRID"),
        (
            Mutation("m1-stage1", M1, "replace", "for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):", "for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=1):"),
            Mutation("m2-stage2", M2, "replace", "for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):", "for kk in tl.range(0, tl.cdiv(k, BLOCK_SIZE_K), num_stages=2):"),
            Mutation("m3-output-init", M3, "insert_before", "tl.store(c_ptrs, c)", "tl.store(c_ptrs, 0.0)"),
            Mutation("m4-output-dup", M4, "duplicate", "tl.store(c_ptrs, c)"),
        ) + triton_fault_mutations("triton-grouped-gemm"),
    ),
    KernelDefinition(
        "tilelang-grouped-gemm", "GROUPED_GEMM", "TileLang", "tilelang", "tilelang-2c31e071", "6692b5edf361d106",
        ("PIPELINED", "TMA", "MMA", "IRREGULAR_GRID"),
        (
            Mutation("m1-stage1", M1, "replace", "num_stages=num_stages", "num_stages=1"),
            Mutation("m2-stage3", M2, "replace", "num_stages=num_stages", "num_stages=3"),
            Mutation("m3-acc-init", M3, "duplicate", "T.clear(C_local)"),
            Mutation("m4-a-init", M4, "insert_before", "T.copy(A[m_start", "T.clear(A_shared)"),
            Mutation("m5-b-init", M5, "insert_before", "if trans_b:", "T.clear(B_shared)"),
            Mutation("m6-a-before", M6, "duplicate", "T.copy(A[m_start"),
            Mutation("m7-a-after", M7, "duplicate_after", "T.copy(A[m_start", target="T.gemm(A_shared"),
        ),
    ),
    KernelDefinition(
        "triton-fused-attention", "FUSED_ATTENTION", "Triton", "triton", "triton-61c02bcd", "f2cb765f0f06e2d0",
        ("PIPELINED", "TMA", "MMA", "ONLINE_SOFTMAX"),
        (
            Mutation("m1-stage1", M1, "replace", "for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):", "for start_n in tl.range(lo, hi, BLOCK_N, num_stages=1, warp_specialize=warp_specialize):"),
            Mutation("m2-stage2", M2, "replace", "for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):", "for start_n in tl.range(lo, hi, BLOCK_N, num_stages=2, warp_specialize=warp_specialize):"),
            Mutation("m3-output-init", M3, "insert_before", "desc_o.store([qo_offset_y, 0], acc.to(dtype))", "desc_o.store([qo_offset_y, 0], tl.zeros([BLOCK_M, HEAD_DIM], dtype=dtype))"),
            Mutation("m4-output-dup", M4, "duplicate", "desc_o.store([qo_offset_y, 0], acc.to(dtype))"),
        ) + triton_fault_mutations("triton-fused-attention"),
    ),
    KernelDefinition(
        "tilelang-fused-attention", "FUSED_ATTENTION", "TileLang", "tilelang", "tilelang-2c31e071", "8d170fca637a486f",
        ("PIPELINED", "TMA", "MMA", "ONLINE_SOFTMAX"),
        (
            Mutation("m1-stage1", M1, "replace", "num_stages=num_stages", "num_stages=1"),
            Mutation("m2-stage3", M2, "replace", "num_stages=num_stages", "num_stages=3"),
            Mutation("m3-acc-init", M3, "duplicate", "T.fill(acc_o, 0)"),
            Mutation("m4-k-init", M4, "insert_before", "T.copy(K[bz,", "T.clear(K_shared)"),
            Mutation("m5-v-init", M5, "insert_before", "T.copy(V[bz,", "T.clear(V_shared)"),
            Mutation("m6-k-before", M6, "duplicate", "T.copy(K[bz,"),
            Mutation("m7-k-after", M7, "duplicate_after", "T.copy(K[bz,", target="T.gemm(Q_shared"),
        ),
    ),
    KernelDefinition(
        "triton-layer-norm", "LAYER_NORM", "Triton", "triton", "triton-61c02bcd", "3a2b5d012ec338da",
        ("REDUCTION", "MULTI_PASS", "FUSED_EPILOGUE"),
        (
            Mutation("m1-stage1", M1, "replace_all", "for off in range(0, N, BLOCK_SIZE):", "for off in tl.range(0, N, BLOCK_SIZE, num_stages=1):"),
            Mutation("m2-stage2", M2, "replace_all", "for off in range(0, N, BLOCK_SIZE):", "for off in tl.range(0, N, BLOCK_SIZE, num_stages=2):"),
            Mutation("m3-output-init", M3, "insert_before", "tl.store(Y + cols, y, mask=mask)", "tl.store(Y + cols, 0.0, mask=mask)"),
            Mutation("m4-output-dup", M4, "duplicate", "tl.store(Y + cols, y, mask=mask)"),
        ) + triton_fault_mutations("triton-layer-norm"),
    ),
    KernelDefinition(
        "tilelang-layer-norm", "LAYER_NORM", "TileLang", "tilelang", "tilelang-2c31e071", "bbd778951e0abc6a",
        ("REDUCTION", "SHARED_MEMORY", "FUSED_EPILOGUE"),
        (
            Mutation("m3-output-init", M3, "insert_before", "T.copy(X_smem, Y[", "T.clear(Y[bx * blk_m, 0])"),
            Mutation("m4-x-init", M4, "insert_before", "T.copy(X[bx * blk_m", "T.clear(X_smem)"),
            Mutation("m5-gamma-init", M5, "insert_before", "T.copy(gamma, G_smem)", "T.clear(G_smem)"),
            Mutation("m6-x-before", M6, "duplicate", "T.copy(X[bx * blk_m"),
            Mutation("m7-x-after", M7, "duplicate_after", "T.copy(X[bx * blk_m", target="T.reduce_sum(X_sq_local"),
            Mutation("m7-output-dup", M7, "duplicate", "T.copy(X_smem, Y["),
        ),
    ),
    KernelDefinition(
        "triton-softmax", "SOFTMAX", "Triton", "triton", "triton-61c02bcd", "908056021440cfc6",
        ("PIPELINED", "REDUCTION", "FUSED_EPILOGUE"),
        (
            Mutation("m1-stage1", M1, "replace", "num_stages=num_stages", "num_stages=1"),
            Mutation("m2-stage3", M2, "replace", "num_stages=num_stages", "num_stages=3"),
            Mutation("m3-output-init", M3, "insert_before", "tl.store(output_ptrs, softmax_output, mask=mask)", "tl.store(output_ptrs, 0.0, mask=mask)"),
            Mutation("m4-output-dup", M4, "duplicate", "tl.store(output_ptrs, softmax_output, mask=mask)"),
        ) + triton_fault_mutations("triton-softmax"),
    ),
    KernelDefinition(
        "tilelang-softmax", "SOFTMAX", "TileLang", "tilelang", "tilelang-2c31e071", "e38d28b2c3f12e15",
        ("PIPELINED", "REDUCTION", "FUSED_EPILOGUE"),
        (
            Mutation("m3-lse-init", M3, "duplicate", "T.fill(lse, -T.infinity(accum_dtype))"),
            Mutation("m4-x-init", M4, "insert_before", "T.copy(X[i_m * BLOCK_M", "T.clear(x)", occurrence=1),
            Mutation("m5-y-init", M5, "insert_before", "T.copy(y, Y[", "T.clear(Y[i_m * BLOCK_M, i_n * BLOCK_N])"),
            Mutation("m6-x-before", M6, "duplicate", "T.copy(X[i_m * BLOCK_M", occurrence=1),
            Mutation("m7-x-after", M7, "duplicate_after", "T.copy(X[i_m * BLOCK_M", occurrence=1, target="T.reduce_max(x, max_x"),
        ),
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, pretty_json(value))


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _candidate(corpus_root: Path, definition: KernelDefinition) -> tuple[dict[str, Any], dict[str, Any]]:
    index = json.loads((corpus_root / definition.source_id / "source-index.json").read_text())
    candidate = next((item for item in index["candidates"] if item["candidate_id"] == definition.candidate_id), None)
    if candidate is None:
        raise RuntimeError(f"missing candidate {definition.candidate_id}")
    manifest = json.loads((corpus_root / candidate["manifest_path"]).read_text())
    return candidate, manifest


def _adapt_source(definition: KernelDefinition, source: str) -> tuple[str, str]:
    changes: list[str] = []
    if definition.kernel_id == "tilelang-fused-attention":
        marker = "@autotune(configs=get_configs(), warmup=10, rep=10)\n"
        if source.count(marker) != 1:
            raise RuntimeError("TileLang attention autotune adapter anchor mismatch")
        source = source.replace(marker, "")
        changes.append("remove the external autotune wrapper while preserving the selected TileLang kernel body")
    if definition.kernel_id == "tilelang-softmax":
        marker = "def pipeguard_resolve(**pipeguard_bindings):\n    return softmax_kernel\n"
        replacement = "def pipeguard_resolve(**pipeguard_bindings):\n    global M, N\n    M = pipeguard_bindings['M']\n    N = pipeguard_bindings['N']\n    return softmax_kernel\n"
        if source.count(marker) != 1:
            raise RuntimeError("TileLang softmax shape adapter anchor mismatch")
        source = source.replace(marker, replacement)
        changes.append("bind the official module-level M and N constants from each Suite 4 workload")
    descriptor = {"kernel_id": definition.kernel_id, "changes": changes or ["none"], "version": 2}
    return source, sha256_text(canonical_json(descriptor))


def _matching_statements(source: str, marker: str) -> list[ast.stmt]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    result: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "end_lineno"):
            continue
        segment = "".join(lines[node.lineno - 1:node.end_lineno])
        if marker in segment:
            result.append(node)
    minimal: list[ast.stmt] = []
    for node in sorted(result, key=lambda item: (item.end_lineno - item.lineno, item.lineno, item.col_offset)):
        if not any(existing.lineno >= node.lineno and existing.end_lineno <= node.end_lineno for existing in minimal):
            minimal.append(node)
    return sorted(minimal, key=lambda item: (item.lineno, item.col_offset))


def _selected_statement(source: str, marker: str, occurrence: int, suffix: str) -> ast.stmt:
    nodes = _matching_statements(source, marker)
    if len(nodes) < occurrence:
        raise RuntimeError(f"{suffix}: marker {marker!r} matched {len(nodes)} statements")
    return nodes[occurrence - 1]


def _statement_edit(source: str, mutation: Mutation) -> str:
    node = _selected_statement(source, mutation.marker, mutation.occurrence, mutation.suffix)
    lines = source.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    segment = "".join(lines[start:end])
    indent = " " * node.col_offset
    if mutation.operation == "duplicate":
        replacement = segment + segment
        return "".join(lines[:start]) + replacement + "".join(lines[end:])
    if mutation.operation == "insert_before":
        replacement = f"{indent}{mutation.value}\n" + segment
        return "".join(lines[:start]) + replacement + "".join(lines[end:])
    if mutation.operation == "insert_after":
        replacement = segment + f"{indent}{mutation.value}\n"
        return "".join(lines[:start]) + replacement + "".join(lines[end:])
    if mutation.operation == "duplicate_after":
        if mutation.target is None:
            raise RuntimeError(f"{mutation.suffix}: duplicate_after lacks target")
        target = _selected_statement(source, mutation.target, mutation.target_occurrence, mutation.suffix)
        if target.end_lineno < node.end_lineno:
            raise RuntimeError(f"{mutation.suffix}: target precedes producer")
        target_indent = " " * target.col_offset
        dedented = "\n".join(line[node.col_offset:] for line in segment.rstrip("\n").splitlines())
        inserted = "\n".join(target_indent + line if line else line for line in dedented.splitlines()) + "\n"
        return "".join(lines[:target.end_lineno]) + inserted + "".join(lines[target.end_lineno:])
    raise ValueError(mutation.operation)


def apply_mutation(source: str, mutation: Mutation, kernel_id: str = "") -> str:
    if mutation.operation == "inject_canary":
        result = inject_triton_fault(source, kernel_id, mutation.suffix)
    elif mutation.operation in {"replace", "replace_all"}:
        count = source.count(mutation.marker)
        expected = 1 if mutation.operation == "replace" else count
        if count < 1 or (mutation.operation == "replace" and count != expected):
            raise RuntimeError(f"{mutation.suffix}: exact marker matched {count} times")
        if mutation.value is None:
            raise RuntimeError(f"{mutation.suffix}: replacement missing")
        result = source.replace(mutation.marker, mutation.value, 1 if mutation.operation == "replace" else -1)
    else:
        result = _statement_edit(source, mutation)
    ast.parse(result)
    if result == source:
        raise RuntimeError(f"{mutation.suffix}: mutation produced no source change")
    return result


def ast_hash(source: str) -> str:
    return sha256_text(ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False))


def _patch(original_name: str, variant_name: str, original: str, variant: str) -> str:
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True), variant.splitlines(keepends=True),
        fromfile=original_name, tofile=variant_name,
    ))


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
        return [] if left == right else [{"path": pointer, "before": left, "after": right}]
    return [] if left == right else [{"path": pointer, "before": left, "after": right}]


def _derived_matrix(definition: KernelDefinition) -> list[dict[str, Any]]:
    family = definition.operator_family
    if family == "DENSE_GEMM":
        rows = [
            {"M": 512, "N": 512, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"M": 256, "N": 512, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"M": 512, "N": 256, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"M": 512, "N": 512, "K": 512, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
        ]
    elif family == "GROUPED_GEMM":
        rows = [
            {"group_sizes": [128, 256], "N": 256, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"group_sizes": [64, 128, 256], "N": 256, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"group_sizes": [128, 256], "N": 128, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
            {"group_sizes": [128, 256], "N": 256, "K": 512, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "dtype": "float16"},
        ]
    elif family == "FUSED_ATTENTION":
        rows = [
            {"batch": 1, "heads": 4, "seq_q": 128, "seq_kv": 128, "dim": 64, "dtype": "float16", "is_causal": True, "BLOCK_M": 64, "BLOCK_N": 64},
            {"batch": 2, "heads": 4, "seq_q": 128, "seq_kv": 128, "dim": 64, "dtype": "float16", "is_causal": True, "BLOCK_M": 64, "BLOCK_N": 64},
            {"batch": 1, "heads": 8, "seq_q": 128, "seq_kv": 128, "dim": 64, "dtype": "float16", "is_causal": True, "BLOCK_M": 64, "BLOCK_N": 64},
            {"batch": 1, "heads": 4, "seq_q": 256, "seq_kv": 256, "dim": 64, "dtype": "float16", "is_causal": False, "BLOCK_M": 64, "BLOCK_N": 64},
        ]
    elif family == "LAYER_NORM":
        rows = [
            {"M": 256, "N": 1024, "BLOCK_M": 1, "dtype": "float16", "eps": 1e-5},
            {"M": 128, "N": 1024, "BLOCK_M": 1, "dtype": "float16", "eps": 1e-5},
            {"M": 256, "N": 2048, "BLOCK_M": 1, "dtype": "float16", "eps": 1e-5},
            {"M": 256, "N": 1024, "BLOCK_M": 2, "dtype": "float16", "eps": 1e-6},
        ]
    elif family == "SOFTMAX":
        rows = [
            {"M": 256, "N": 1024, "BLOCK_M": 1, "BLOCK_N": 1024, "dtype": "float16"},
            {"M": 128, "N": 1024, "BLOCK_M": 1, "BLOCK_N": 1024, "dtype": "float16"},
            {"M": 256, "N": 2048, "BLOCK_M": 1, "BLOCK_N": 2048, "dtype": "float16"},
            {"M": 256, "N": 1024, "BLOCK_M": 2, "BLOCK_N": 512, "dtype": "float16"},
        ]
    else:
        raise AssertionError(family)
    return [{"role": "ORIGINAL" if index == 0 else "BOUNDARY", "bindings": row} for index, row in enumerate(rows)]


def _triton_compiler(
    grid: list[int], signature: dict[str, str], constexprs: dict[str, Any],
    pointers: dict[str, str], *, num_warps: int = 4, num_stages: int = 3,
) -> dict[str, Any]:
    arg_names = list(signature)
    pointer_allocations = {str(arg_names.index(name)): allocation for name, allocation in pointers.items()}
    return {
        "grid": grid,
        "grid_ctas": math.prod(grid),
        "selected_cta": [0, 0, 0],
        "attrs": {},
        "options": {"num_ctas": 1, "num_warps": num_warps, "num_stages": num_stages},
        "signature": signature,
        "constexprs": constexprs,
        "binding_evidence": {
            "evidence_version": "triton-binding-evidence-v1",
            "pointer_allocations": pointer_allocations,
            "scalar_values": {},
            "unit_attributes": {},
            "array_attributes": {},
        },
    }


def _materialization(definition: KernelDefinition, b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = {"backend": "cuda", "arch": "sm90" if definition.dsl == "triton" else "sm_90a", "warp_size": 32}
    resolver: dict[str, Any] = {}
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    compiler: dict[str, Any]
    if definition.dsl == "triton" and definition.operator_family == "DENSE_GEMM":
        m, n, k = b["M"], b["N"], b["K"]
        bm, bn, bk = b["BLOCK_M"], b["BLOCK_N"], b["BLOCK_K"]
        signature = {
            "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
            "M": "constexpr", "N": "constexpr", "K": "constexpr",
            "stride_am": "constexpr", "stride_ak": "constexpr", "stride_bk": "constexpr", "stride_bn": "constexpr",
            "stride_cm": "constexpr", "stride_cn": "constexpr", "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_K": "constexpr", "GROUP_SIZE_M": "constexpr", "ACTIVATION": "constexpr",
        }
        constexprs = {
            "M": m, "N": n, "K": k, "stride_am": k, "stride_ak": 1, "stride_bk": n, "stride_bn": 1,
            "stride_cm": n, "stride_cn": 1, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": 8, "ACTIVATION": "",
        }
        compiler = _triton_compiler([math.ceil(m / bm) * math.ceil(n / bn), 1, 1], signature, constexprs, {"a_ptr": "global:a", "b_ptr": "global:b", "c_ptr": "global:c"})
        numeric = {"oracle": "torch.matmul", "seed": 17, "atol": 0.02, "rtol": 0.02, "timeout_seconds": 30.0}
    elif definition.dsl == "tilelang" and definition.operator_family == "DENSE_GEMM":
        kwargs = {"M": b["M"], "N": b["N"], "K": b["K"], "block_M": b["BLOCK_M"], "block_N": b["BLOCK_N"], "block_K": b["BLOCK_K"]}
        compiler = {"grid": [math.ceil(b["N"] / b["BLOCK_N"]), math.ceil(b["M"] / b["BLOCK_M"]), 1], "selected_cta": [0, 0, 0], "pass_configs": {}}
        numeric = {"oracle": "torch.matmul", "seed": 19, "atol": 0.02, "rtol": 0.02, "timeout_seconds": 30.0}
    elif definition.dsl == "triton" and definition.operator_family == "GROUPED_GEMM":
        groups = len(b["group_sizes"])
        grid = [8, 1, 1]
        signature = {
            "group_a_ptrs": "*u64", "group_b_ptrs": "*u64", "group_c_ptrs": "*u64",
            "group_gemm_sizes": "*i32", "g_lds": "*i32", "group_size": "i32",
            "NUM_SM": "constexpr", "BLOCK_SIZE_M": "constexpr", "BLOCK_SIZE_N": "constexpr", "BLOCK_SIZE_K": "constexpr",
        }
        constexprs = {"NUM_SM": grid[0], "BLOCK_SIZE_M": b["BLOCK_M"], "BLOCK_SIZE_N": b["BLOCK_N"], "BLOCK_SIZE_K": b["BLOCK_K"]}
        compiler = _triton_compiler(grid, signature, constexprs, {
            "group_a_ptrs": "global:a_ptrs", "group_b_ptrs": "global:b_ptrs", "group_c_ptrs": "global:c_ptrs",
            "group_gemm_sizes": "global:sizes", "g_lds": "global:lds",
        })
        numeric = {"oracle": "per-group torch.matmul", "seed": 23, "atol": 0.03, "rtol": 0.03, "timeout_seconds": 45.0}
    elif definition.dsl == "tilelang" and definition.operator_family == "GROUPED_GEMM":
        kwargs = {
            "batch_sizes_list": b["group_sizes"], "K": b["K"], "N": b["N"], "block_M": b["BLOCK_M"],
            "block_N": b["BLOCK_N"], "block_K": b["BLOCK_K"], "num_stages": 2, "threads": 128,
        }
        compiler = {
            "grid": [sum(math.ceil(size / b["BLOCK_M"]) for size in b["group_sizes"]), math.ceil(b["N"] / b["BLOCK_N"]), 1],
            "selected_cta": [0, 0, 0], "pass_configs": {}, "tuple_materialization_kwargs": ["batch_sizes_list"],
        }
        numeric = {"oracle": "per-group torch.matmul", "seed": 29, "atol": 0.03, "rtol": 0.03, "timeout_seconds": 45.0}
    elif definition.dsl == "triton" and definition.operator_family == "FUSED_ATTENTION":
        block_m, block_n = b["BLOCK_M"], b["BLOCK_N"]
        signature = {
            "sm_scale": "fp32", "M": "*fp32", "Z": "i32", "H": "i32", "desc_q": "*fp16", "desc_k": "*fp16",
            "desc_v": "*fp16", "desc_o": "*fp16", "N_CTX": "i32", "HEAD_DIM": "constexpr", "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr", "FP8_OUTPUT": "constexpr", "STAGE": "constexpr", "warp_specialize": "constexpr", "IS_HOPPER": "constexpr",
        }
        constexprs = {"HEAD_DIM": b["dim"], "BLOCK_M": block_m, "BLOCK_N": block_n, "FP8_OUTPUT": False, "STAGE": 3 if b["is_causal"] else 1, "warp_specialize": False, "IS_HOPPER": True}
        compiler = _triton_compiler([math.ceil(b["seq_q"] / block_m), b["batch"] * b["heads"], 1], signature, constexprs, {
            "M": "global:m", "desc_q": "global:q", "desc_k": "global:k", "desc_v": "global:v", "desc_o": "global:o",
        }, num_warps=4, num_stages=3)
        numeric = {"oracle": "torch.scaled_dot_product_attention", "seed": 31, "atol": 0.03, "rtol": 0.03, "timeout_seconds": 60.0}
    elif definition.dsl == "tilelang" and definition.operator_family == "FUSED_ATTENTION":
        args = [b["batch"], b["heads"], b["seq_q"], b["seq_kv"], b["dim"], b["is_causal"], b["BLOCK_M"], b["BLOCK_N"], 2, 128]
        compiler = {"grid": [math.ceil(b["seq_q"] / b["BLOCK_M"]), b["heads"], b["batch"]], "selected_cta": [0, 0, 0], "pass_configs": {}}
        numeric = {"oracle": "torch.scaled_dot_product_attention", "seed": 37, "atol": 0.03, "rtol": 0.03, "timeout_seconds": 60.0}
    elif definition.dsl == "triton" and definition.operator_family == "LAYER_NORM":
        block = 1 << (b["N"] - 1).bit_length()
        signature = {"X": "*fp16", "Y": "*fp16", "W": "*fp16", "B": "*fp16", "Mean": "*fp32", "Rstd": "*fp32", "stride": "i32", "N": "i32", "eps": "fp32", "BLOCK_SIZE": "constexpr"}
        compiler = _triton_compiler([b["M"], 1, 1], signature, {"BLOCK_SIZE": block}, {"X": "global:x", "Y": "global:y", "W": "global:w", "B": "global:b", "Mean": "global:mean", "Rstd": "global:rstd"}, num_warps=4, num_stages=3)
        numeric = {"oracle": "torch.nn.functional.layer_norm", "seed": 41, "atol": 0.02, "rtol": 0.02, "timeout_seconds": 30.0}
    elif definition.dsl == "tilelang" and definition.operator_family == "LAYER_NORM":
        args = [b["M"], b["N"]]
        kwargs = {"eps": b["eps"], "blk_m": b["BLOCK_M"], "threads": 256, "in_dtype": b["dtype"], "out_dtype": b["dtype"]}
        compiler = {"grid": [math.ceil(b["M"] / b["BLOCK_M"]), 1, 1], "selected_cta": [0, 0, 0], "pass_configs": {}}
        numeric = {"oracle": "torch.nn.functional.layer_norm", "seed": 43, "atol": 0.03, "rtol": 0.03, "timeout_seconds": 30.0}
    elif definition.dsl == "triton" and definition.operator_family == "SOFTMAX":
        block = 1 << (b["N"] - 1).bit_length()
        stages = 2
        signature = {"output_ptr": "*fp16", "input_ptr": "*fp16", "input_row_stride": "i32", "output_row_stride": "i32", "n_rows": "i32", "n_cols": "i32", "BLOCK_SIZE": "constexpr", "num_stages": "constexpr"}
        compiler = _triton_compiler([b["M"], 1, 1], signature, {"BLOCK_SIZE": block, "num_stages": stages}, {"output_ptr": "global:output", "input_ptr": "global:input"}, num_warps=4, num_stages=stages)
        numeric = {"oracle": "torch.softmax", "seed": 47, "atol": 0.002, "rtol": 0.002, "timeout_seconds": 30.0}
    elif definition.dsl == "tilelang" and definition.operator_family == "SOFTMAX":
        resolver = {"M": b["M"], "N": b["N"]}
        kwargs = {"BLOCK_M": b["BLOCK_M"], "BLOCK_N": b["BLOCK_N"], "dtype": b["dtype"]}
        compiler = {"grid": [math.ceil(b["M"] / b["BLOCK_M"]), 1, 1], "selected_cta": [0, 0, 0], "pass_configs": {}}
        numeric = {"oracle": "torch.softmax", "seed": 53, "atol": 0.003, "rtol": 0.003, "timeout_seconds": 45.0}
    else:
        raise AssertionError(definition.kernel_id)
    return {"resolver_kwargs": resolver, "args": args, "kwargs": kwargs}, compiler, {**numeric, "target": target}


def _workloads(repository_root: Path, definition: KernelDefinition, manifest: dict[str, Any], checkout: Path) -> list[dict[str, Any]]:
    matrix = _derived_matrix(definition)
    original_bindings = matrix[0]["bindings"]
    result: list[dict[str, Any]] = []
    original_id = f"{definition.kernel_id}-workload-original"
    for index, item in enumerate(matrix):
        role = item["role"]
        spec_id = original_id if role == "ORIGINAL" else f"{definition.kernel_id}-workload-boundary-{index}"
        bindings = item["bindings"]
        materialization, compiler, numeric_with_target = _materialization(definition, bindings)
        target = numeric_with_target.pop("target")
        recipe = {
            "schema_version": "pipeguard-mutation-workload-v1", "recipe_id": spec_id,
            "kernel_id": definition.kernel_id, "dsl": definition.dsl, "bindings": bindings,
            "materialization": materialization, "compiler": compiler, "numeric": numeric_with_target, "target": target,
        }
        recipe_rel = f"corpus/mutations/suite4/kernels/{definition.kernel_id}/recipes/{spec_id}.json"
        recipe_path = repository_root / recipe_rel
        write_json(recipe_path, recipe)
        evidence_path = manifest["upstream"]["relative_path"]
        delta = [] if role == "ORIGINAL" else _diff_bindings(original_bindings, bindings)
        dimensions = [] if role == "ORIGINAL" else sorted({entry["path"].split("/")[2] for entry in delta})
        result.append({
            "spec_id": spec_id, "kernel_id": definition.kernel_id, "role": role,
            **({} if role == "ORIGINAL" else {"parent_spec_id": original_id}),
            "bindings": bindings, "delta": delta, "boundary_dimensions": dimensions,
            "evidence": {
                "kind": "UPSTREAM_TEST" if role == "ORIGINAL" else "SUITE4_DERIVED_BOUNDARY",
                "path": evidence_path, "sha256": sha256_file(checkout / evidence_path),
                "rationale": "Pinned official example kernel with an independent PyTorch oracle." if role == "ORIGINAL" else "Controlled Suite 4 boundary derived from the paired official operator and validated by the same independent PyTorch oracle.",
            },
            "recipe": {"path": recipe_rel, "sha256": sha256_file(recipe_path)},
        })
    return result


def _adjudication(mutation: Mutation, kernel_id: str) -> dict[str, str]:
    if kernel_id == "tilelang-dense-gemm" and mutation.suffix == "m6-a-before":
        return {
            "class": "DATA_HAZARD",
            "status": "CONFIRMED",
            "method": "MANUAL_PROTOCOL_TRACE",
            "rationale": (
                "The duplicated A producer lowers to two overlapping TMA writes under the same "
                "mbarrier lifecycle and generation without a successful wait or typed ordering "
                "relation between their asynchronous windows."
            ),
        }
    if mutation.operation == "inject_canary":
        strategy = TRITON_FAULT_STRATEGIES[mutation.suffix]
        if kernel_id in strategy.rejected_kernel_ids:
            return {
                "class": "UNKNOWN",
                "status": "REJECTED",
                "method": "MANUAL_PROTOCOL_TRACE",
                "rationale": strategy.rejection_rationale
                or "The prior mutation label was rejected by final-lowering audit.",
            }
        return {
            "class": strategy.adjudication_class,
            "status": strategy.adjudication_status,
            "method": strategy.adjudication_method,
            "rationale": strategy.adjudication_rationale,
        }
    return {
        "class": "UNKNOWN", "status": "UNADJUDICATED", "method": "UNADJUDICATED",
        "rationale": "The mutation preserves intended arithmetic but requires independent protocol adjudication; checker output is not used as ground truth.",
    }


def generate(repository_root: Path, checkout_parent: Path) -> dict[str, int]:
    repository_root = repository_root.resolve(strict=True)
    corpus_root = repository_root / "corpus" / "upstream"
    suite_root = repository_root / "corpus" / "mutations" / "suite4"
    suite_root.mkdir(parents=True, exist_ok=True)
    baseline_path = suite_root / "baseline-index.json"
    catalog_path = suite_root / "catalog.json"
    previous_baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    previous_catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
    checkouts = {"triton-61c02bcd": checkout_parent / "triton", "tilelang-2c31e071": checkout_parent / "tilelang"}
    suite3_lock_path = repository_root / "corpus/upstream/suite3/lock.json"
    suite3_catalog_path = repository_root / "corpus/upstream/suite3/catalog.json"
    suite3_lock = json.loads(suite3_lock_path.read_text())
    repositories = {item["source_id"]: item for item in suite3_lock["repositories"]}
    for source_id, checkout in checkouts.items():
        if git_commit(checkout) != repositories[source_id]["commit"]:
            raise RuntimeError(f"{source_id} checkout commit mismatch")

    rules_path = suite_root / "mutation-rules.json"
    write_json(rules_path, {"schema_version": "pipeguard-mutation-rules-v1", "rules": RULES})
    kernel_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    workload_records: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    expected_directories = {definition.kernel_id for definition in KERNELS}
    kernels_root = suite_root / "kernels"
    if kernels_root.exists():
        for child in kernels_root.iterdir():
            if child.is_dir() and child.name not in expected_directories:
                shutil.rmtree(child)

    for definition in KERNELS:
        candidate, manifest = _candidate(corpus_root, definition)
        capsule_path = corpus_root / candidate["capsule_path"]
        if sha256_file(capsule_path) != manifest["capsule"]["sha256"]:
            raise RuntimeError(f"{definition.kernel_id} capsule hash mismatch")
        original, adapter_hash = _adapt_source(definition, capsule_path.read_text(encoding="utf-8"))
        kernel_directory = kernels_root / definition.kernel_id
        if kernel_directory.exists():
            shutil.rmtree(kernel_directory)
        source_dir = kernel_directory / "sources"
        patch_dir = kernel_directory / "patches"
        original_path = source_dir / "original.py"
        write_text(original_path, original)
        original_variant_id = f"{definition.kernel_id}-source-original"
        variants = [{
            "variant_id": original_variant_id, "kernel_id": definition.kernel_id, "role": "ORIGINAL",
            "source": {"path": str(original_path.relative_to(repository_root)), "sha256": sha256_file(original_path)},
            "adapter_sha256": adapter_hash, "selected_ast_sha256": ast_hash(original),
            "expected_computational_semantics": "PRESERVE_OUTPUT",
            "adjudication": {"class": "CORRECT", "status": "CONFIRMED", "method": "REFERENCE_ORACLE", "rationale": "Pinned official source replayed successfully against its independent PyTorch reference on the original and all three boundary workloads."},
        }]
        for mutation in definition.mutations:
            mutated = apply_mutation(original, mutation, definition.kernel_id)
            variant_id = f"{definition.kernel_id}-source-{mutation.suffix}"
            variant_path = source_dir / f"{mutation.suffix}.py"
            patch_path = patch_dir / f"{mutation.suffix}.diff"
            write_text(variant_path, mutated)
            write_text(patch_path, _patch("original.py", f"{mutation.suffix}.py", original, mutated))
            rule = RULE_BY_ID[mutation.rule_id]
            variants.append({
                "variant_id": variant_id, "kernel_id": definition.kernel_id, "role": "MUTANT",
                "parent_variant_id": original_variant_id, "mutation_rule_id": mutation.rule_id,
                "mutation_family": rule["family"], "operation": mutation.operation,
                "anchor_sha256": sha256_text(original if mutation.operation == "inject_canary" else mutation.marker),
                "replacement_sha256": sha256_text(mutated if mutation.operation == "inject_canary" else mutation.value or mutation.target or ""),
                "occurrence": mutation.occurrence,
                "source": {"path": str(variant_path.relative_to(repository_root)), "sha256": sha256_file(variant_path)},
                "patch": {"path": str(patch_path.relative_to(repository_root)), "sha256": sha256_file(patch_path)},
                "adapter_sha256": adapter_hash, "selected_ast_sha256": ast_hash(mutated),
                "expected_computational_semantics": "PRESERVE_OUTPUT",
                "adjudication": _adjudication(mutation, definition.kernel_id),
            })
        workloads = _workloads(repository_root, definition, manifest, checkouts[definition.source_id])
        variant_records.extend(variants)
        workload_records.extend(workloads)
        for variant in variants:
            for workload in workloads:
                case_records.append({
                    "case_id": f"{variant['variant_id']}--{workload['spec_id']}", "kernel_id": definition.kernel_id,
                    "variant_id": variant["variant_id"], "spec_id": workload["spec_id"], "selected_cta": [0, 0, 0],
                    "pipeline_status": "UNASSESSED", "numeric_status": "NOT_RUN", "checker_status": "NOT_RUN",
                })
        kernel_records.append({
            "kernel_id": definition.kernel_id, "operator_family": definition.operator_family,
            "family": definition.operator_family, "project": definition.project, "dsl": definition.dsl,
            "source_id": definition.source_id, "candidate_id": definition.candidate_id,
            "path": manifest["upstream"]["relative_path"], "qualname": manifest["upstream"]["qualname"],
            "features": list(definition.features), "source_sha256": manifest["upstream"]["source_sha256"],
            "selected_source_sha256": manifest["upstream"]["selected_source_sha256"], "capsule_sha256": manifest["capsule"]["sha256"],
            "variant_ids": [item["variant_id"] for item in variants], "workload_spec_ids": [item["spec_id"] for item in workloads],
        })
        write_json(kernel_directory / "manifest.json", {"kernel": kernel_records[-1], "variants": variants, "workloads": workloads})

    selected_sources = sorted({item.source_id for item in KERNELS} | {item["frontend_source_id"] for item in suite3_lock["toolchains"]})
    lock = {
        "schema_version": "pipeguard-mutation-suite-lock-v1", "suite_id": SUITE_ID, "suite_revision": 2,
        "suite3": {
            "lock": {"path": str(suite3_lock_path.relative_to(repository_root)), "sha256": sha256_file(suite3_lock_path)},
            "catalog": {"path": str(suite3_catalog_path.relative_to(repository_root)), "sha256": sha256_file(suite3_catalog_path)},
        },
        "policy": {
            "operator_family_count": 5, "implementations_per_family": 2, "kernel_count": 10,
            "minimum_mutants": 4, "minimum_mutation_families": 2, "original_workloads": 1,
            "boundary_workloads": 3, "minimum_boundary_dimensions": 2,
        },
        "target": {"backend": "cuda", "architecture": "sm90", "warp_size": 32, "selected_cta": [0, 0, 0]},
        "repositories": [{
            "source_id": source_id, "project": repositories[source_id]["project"], "dsl": repositories[source_id]["dsl"],
            "commit": repositories[source_id]["commit"],
            "source_index": {"path": f"corpus/upstream/{source_id}/source-index.json", "sha256": sha256_file(corpus_root / source_id / "source-index.json")},
        } for source_id in selected_sources],
        "toolchains": [item for item in suite3_lock["toolchains"] if item["dsl"] in {"tilelang", "triton"}],
        "mutation_rules": {"path": str(rules_path.relative_to(repository_root)), "sha256": sha256_file(rules_path)},
    }
    lock_path = suite_root / "lock.json"
    write_json(lock_path, lock)
    catalog = {
        "schema_version": "pipeguard-mutation-catalog-v1", "suite_id": SUITE_ID,
        "suite_lock_sha256": sha256_file(lock_path), "kernels": sorted(kernel_records, key=lambda item: item["kernel_id"]),
        "variants": sorted(variant_records, key=lambda item: item["variant_id"]),
        "workloads": sorted(workload_records, key=lambda item: item["spec_id"]),
        "cases": sorted(case_records, key=lambda item: item["case_id"]),
    }
    catalog_path = suite_root / "catalog.json"
    write_json(catalog_path, catalog)
    counts = {
        "kernels": len(kernel_records), "original_variants": sum(item["role"] == "ORIGINAL" for item in variant_records),
        "mutant_variants": sum(item["role"] == "MUTANT" for item in variant_records),
        "original_workloads": sum(item["role"] == "ORIGINAL" for item in workload_records),
        "boundary_workloads": sum(item["role"] == "BOUNDARY" for item in workload_records),
        "cases": len(case_records), "assessed_cases": 0,
    }
    catalog_hash = sha256_file(catalog_path)
    retained_results = []
    if (
        isinstance(previous_baseline, dict)
        and previous_baseline.get("suite_id") == SUITE_ID
        and isinstance(previous_catalog, dict)
        and previous_catalog.get("suite_id") == SUITE_ID
    ):
        old_variants = {item["variant_id"]: item for item in previous_catalog.get("variants", [])}
        old_workloads = {item["spec_id"]: item for item in previous_catalog.get("workloads", [])}
        new_variants = {item["variant_id"]: item for item in variant_records}
        new_workloads = {item["spec_id"]: item for item in workload_records}
        unchanged_cases = {
            item["case_id"]
            for item in case_records
            if old_variants.get(item["variant_id"]) == new_variants[item["variant_id"]]
            and old_workloads.get(item["spec_id"]) == new_workloads[item["spec_id"]]
        }
        retained_results = [
            item for item in previous_baseline.get("results", [])
            if item.get("case_id") in unchanged_cases
        ]
    counts["assessed_cases"] = len(retained_results)
    if retained_results:
        runtime_counts = {
            "lowered_cases": ("lowering_status", "LOWERED"),
            "captured_snapshot_cases": ("capture_status", "CAPTURED"),
            "parsed_pgir_cases": ("pgir_status", "PARSED"),
            "checked_cases": ("checker_status", "CHECKED"),
            "numeric_pass_cases": ("numeric_status", "PASS"),
            "numeric_mismatch_cases": ("numeric_status", "NUMERIC_MISMATCH"),
            "numeric_hang_cases": ("numeric_status", "HANG"),
            "numeric_compile_error_cases": ("numeric_status", "COMPILE_ERROR"),
        }
        counts.update({
            key: sum(item.get(field) == value for item in retained_results)
            for key, (field, value) in runtime_counts.items()
        })
    write_json(baseline_path, {
        "schema_version": "pipeguard-mutation-baseline-index-v1", "suite_id": SUITE_ID,
        "suite_lock_sha256": sha256_file(lock_path), "catalog_sha256": catalog_hash,
        "counts": counts, "results": retained_results,
    })
    return counts


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the deterministic paired-official PipeGuard Suite 4 corpus")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--checkout-parent", type=Path, default=Path("/mnt/project_dft/xuwen/PipeGuard-asplos"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    print(canonical_json(generate(arguments.repository_root, arguments.checkout_parent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
