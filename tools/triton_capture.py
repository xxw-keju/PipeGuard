#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "triton-snapshot-v1"
ADAPTER_VERSION = "triton-pgir-v1"


class CaptureError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def diagnostic(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def _git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


class TritonCapture:
    _DECLARATIONS = {"arith.constant"}
    _CONTAINERS = {"builtin.module", "tt.func", "tt.return"}
    _SUPPORTED = {
        "ttng.init_barrier",
        "ttng.arrive_barrier",
        "ttng.wait_barrier",
    }

    def __init__(self, triton: Any, module: Any, source: str, command: str):
        self.triton = triton
        self.module = module
        self.source = source
        self.command = command
        self.operations: list[Any] = []
        module.walk(self.operations.append)
        functions = [operation for operation in self.operations if operation.get_name() == "tt.func"]
        if len(functions) != 1:
            raise CaptureError(
                "TRITON_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one TTGIR tt.func",
                {"functions": len(functions)},
            )
        self.num_ctas = self._integer_module_attr("ttg.num-ctas")
        self.num_warps = self._integer_module_attr("ttg.num-warps")
        self.threads_per_warp = self._integer_module_attr("ttg.threads-per-warp")
        target = module.get_operation().get_str_attr("ttg.target")
        if target != "cuda:90":
            raise CaptureError(
                "TRITON_TARGET_UNSUPPORTED",
                "Triton capture supports only cuda:90 TTGIR",
                {"target": target},
            )
        if self.num_ctas != 1:
            raise CaptureError(
                "TRITON_MULTI_CTA_UNSUPPORTED",
                "PGIR-v1 Triton capture requires ttg.num-ctas = 1",
                {"num_ctas": self.num_ctas},
            )
        if self.threads_per_warp != 32 or self.num_warps <= 0:
            raise CaptureError(
                "TRITON_PARTICIPANTS_UNRESOLVED",
                "Triton warp configuration is outside the static SM90 subset",
                {"num_warps": self.num_warps, "threads_per_warp": self.threads_per_warp},
            )
        self.actor = "cta"
        self.cta_threads = self.num_warps * self.threads_per_warp
        self.result_definitions: dict[int, tuple[int, int, Any]] = {}
        self.constants: dict[int, Any] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.output_operations: list[dict[str, Any]] = []
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}
        self.sequence = 0

    def _integer_module_attr(self, name: str) -> int:
        value = self.module.get_int_attr(name)
        if value is None:
            raise CaptureError(
                "TRITON_MODULE_ATTRIBUTE_MISSING",
                "TTGIR module lacks a required static integer attribute",
                {"attribute": name},
            )
        return int(value)

    def _record_definitions(self) -> None:
        for index, operation in enumerate(self.operations):
            for result_index in range(operation.get_num_results()):
                result = operation.get_result(result_index)
                self.result_definitions[result.id()] = (index, result_index, result)
            if operation.get_name() == "arith.constant" and operation.get_num_results() == 1:
                try:
                    self.constants[operation.get_result(0).id()] = operation.get_constant_value()
                except RuntimeError:
                    pass

    def _constant(self, value: Any, role: str) -> Any:
        if value.id() not in self.constants:
            raise CaptureError(
                "TRITON_SSA_VALUE_UNRESOLVED",
                "Triton protocol operand is not a static SSA constant",
                {"role": role, "type": str(value.get_type())},
            )
        return self.constants[value.id()]

    def _resource(self, value: Any) -> str:
        definition = self.result_definitions.get(value.id())
        if definition is None:
            raise CaptureError(
                "TRITON_BARRIER_IDENTITY_UNRESOLVED",
                "mbarrier SSA value has no local definition",
            )
        operation_index, result_index, result = definition
        if self.operations[operation_index].get_name() != "ttg.local_alloc":
            raise CaptureError(
                "TRITON_BARRIER_ALIAS_UNSUPPORTED",
                "mbarrier must refer directly to a ttg.local_alloc result",
                {"defining_operation": self.operations[operation_index].get_name()},
            )
        result_type = str(result.get_type())
        if "!ttg.memdesc<" not in result_type or "xi64" not in result_type or "shared_memory" not in result_type:
            raise CaptureError(
                "TRITON_BARRIER_TYPE_UNSUPPORTED",
                "mbarrier allocation type is not a shared i64 memdesc",
                {"type": result_type},
            )
        identifier = f"mbarrier:ssa:{operation_index}:{result_index}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "mbarrier",
            "participants": [self.actor],
        })
        return identifier

    def _emit(self, op: str, operation_index: int, **fields: Any) -> None:
        self.output_operations.append({
            "id": f"triton:{self.actor}:e{self.sequence:05d}:op{operation_index:05d}",
            "op": op,
            "actor": self.actor,
            "seq": self.sequence,
            **fields,
        })
        self.sequence += 1

    def capture(self) -> dict[str, Any]:
        self._record_definitions()
        semantic_blocks: set[int] = set()
        for index, operation in enumerate(self.operations):
            name = operation.get_name()
            if name in self._CONTAINERS:
                continue
            if name == "ttg.local_alloc":
                if operation.get_num_operands() != 0:
                    raise CaptureError(
                        "TRITON_OPERATION_UNSUPPORTED",
                        "ttg.local_alloc with an input has a shared-memory write effect",
                        {"operation": name, "operation_index": index},
                    )
                continue
            if name in self._DECLARATIONS:
                continue
            if name not in self._SUPPORTED:
                raise CaptureError(
                    "TRITON_OPERATION_UNSUPPORTED",
                    "TTGIR operation has no sound capture rule",
                    {"operation": name, "operation_index": index},
                )
            block = operation.get_block()
            if block is None:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNRESOLVED",
                    "protocol operation is not attached to an executable block",
                    {"operation": name, "operation_index": index},
                )
            semantic_blocks.add(block.id())
            if len(semantic_blocks) > 1:
                raise CaptureError(
                    "TRITON_CONTROL_FLOW_UNSUPPORTED",
                    "initial Triton capture supports one straight-line executable block",
                    {"blocks": len(semantic_blocks)},
                )
            if operation.get_num_operands() < 1:
                raise CaptureError(
                    "TRITON_OPERATION_MALFORMED",
                    "mbarrier operation lacks its SSA resource operand",
                    {"operation": name},
                )
            resource = self._resource(operation.get_operand(0))
            if name == "ttng.init_barrier":
                count = operation.get_int_attr("count")
                if count is None or int(count) <= 0:
                    raise CaptureError(
                        "TRITON_MBARRIER_COUNT_UNRESOLVED",
                        "ttng.init_barrier count must be a static positive integer",
                    )
                self._emit(
                    "triton.mbarrier.init",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    count=int(count),
                )
            elif name == "ttng.arrive_barrier":
                count = operation.get_int_attr("count")
                if count is None or int(count) <= 0 or operation.get_num_operands() != 2:
                    raise CaptureError(
                        "TRITON_MBARRIER_ARRIVE_UNRESOLVED",
                        "ttng.arrive_barrier requires static count and predicate",
                    )
                predicate_operand = operation.get_operand(1)
                predicate_value = self._constant(predicate_operand, "arrive predicate")
                if str(predicate_operand.get_type()) != "i1" or predicate_value not in {0, 1, -1}:
                    raise CaptureError(
                        "TRITON_PREDICATE_UNRESOLVED",
                        "ttng.arrive_barrier predicate must be a scalar i1 constant",
                        {"value": str(predicate_value), "type": str(predicate_operand.get_type())},
                    )
                predicate = bool(predicate_value)
                generation = self.arrival_rank.get(resource, 0)
                self.arrival_rank[resource] = generation + 1
                self._emit(
                    "triton.mbarrier.arrive",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    generation=generation,
                    count=int(count),
                    predicate=predicate,
                )
            else:
                if operation.get_num_operands() != 2:
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier requires one scalar phase SSA operand",
                    )
                parity = self._constant(operation.get_operand(1), "wait phase")
                if not isinstance(parity, int) or isinstance(parity, bool):
                    raise CaptureError(
                        "TRITON_WAIT_PHASE_UNRESOLVED",
                        "ttng.wait_barrier phase must be a scalar integer constant",
                        {"value": str(parity)},
                    )
                generation = self.wait_rank.get(resource, 0)
                if parity != generation % 2:
                    raise CaptureError(
                        "TRITON_PARITY_UNRESOLVED",
                        "wait phase SSA disagrees with the recovered gap-free stream rank",
                        {"resource": resource, "rank": generation, "parity": parity},
                    )
                self.wait_rank[resource] = generation + 1
                self._emit(
                    "triton.mbarrier.wait_parity",
                    index,
                    resource=resource,
                    lifecycle_epoch=0,
                    generation=generation,
                    raw_parity=parity,
                    logical_stream_id=f"{self.actor}:{resource}",
                    stream_rank=generation,
                    polling_coalesced=True,
                )
        if not self.output_operations:
            raise CaptureError(
                "TRITON_OPERATIONS_MISSING",
                "TTGIR module contains no supported protocol operations",
            )
        triton_root = Path(self.triton.__file__).resolve().parents[1]
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "producer": {
                "dsl_version": str(self.triton.__version__),
                "dsl_commit": _git_revision(triton_root),
                "backend_commit": "triton-nvidia-ttgir",
                "adapter_version": ADAPTER_VERSION,
                "command": self.command,
            },
            "target": {
                "architecture": "sm90",
                "warp_size": 32,
                "cta_threads": self.cta_threads,
                "shared_memory_bytes": 0,
                "grid_scope": "single_cta",
            },
            "extraction": {
                "complete": True,
                "stage": "post-ttgir-pipeline",
                "source": self.source,
                "finite_control_flow": True,
                "statically_known_participants": True,
                "fair_async_completion": True,
                "no_cluster_effects": True,
                "alias_precision": "EXACT",
                "parity_normalization_status": "RESOLVED",
            },
            "actors": [{
                "id": self.actor,
                "kind": "PARTITION",
                "participants": list(range(self.cta_threads)),
            }],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [],
            "operations": self.output_operations,
            "unsupported_constructs": [],
        }


def capture_file(path: Path, source: str, command: str) -> dict[str, Any]:
    try:
        import triton
        from triton._C.libtriton import ir, nvidia
    except ImportError as error:
        raise CaptureError(
            "TRITON_RUNTIME_UNAVAILABLE",
            "Triton capture must run in the isolated Triton environment",
            {"error": str(error)},
        ) from error
    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    try:
        module = ir.parse_mlir_module(str(path), context)
    except RuntimeError as error:
        raise CaptureError(
            "TRITON_IR_INPUT_INVALID",
            "TTGIR MLIR module could not be parsed",
            {"path": str(path), "error": str(error)},
        ) from error
    return TritonCapture(triton, module, source, command).capture()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a strict Triton TTGIR snapshot for PipeGuard")
    parser.add_argument("input", type=Path, help="one post-pipeline TTGIR MLIR module")
    parser.add_argument("--source", required=True, help="stable source/provenance label")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(args.input, args.source, " ".join(sys.argv))
    except CaptureError as error:
        print(json.dumps({
            "captured": False,
            "verdict": "UNSUPPORTED",
            "diagnostics": [error.diagnostic()],
        }, sort_keys=True, separators=(",", ":")))
        return 2
    options = {"sort_keys": True, "ensure_ascii": False}
    if args.pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    print(json.dumps(result, **options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
