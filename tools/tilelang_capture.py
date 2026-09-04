#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "tilelang-snapshot-v2"
ADAPTER_VERSION = "tilelang-pgir-v2"


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


def _stable_path(path: tuple[Any, ...]) -> str:
    encoded = "/".join(map(str, path)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


class TileLangCapture:
    def __init__(
        self,
        tvm: Any,
        tirx: Any,
        module: Any,
        source: str,
        command: str,
        source_ir_sha256: str,
    ):
        self.tvm = tvm
        self.tirx = tirx
        self.module = module
        self.source = source
        self.command = command
        self.source_ir_sha256 = source_ir_sha256
        functions = [func for func in module.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(functions) != 1:
            raise CaptureError(
                "TILELANG_KERNEL_COUNT_UNSUPPORTED",
                "capture requires exactly one lowered TileLang PrimFunc",
                {"primfuncs": len(functions)},
            )
        self.func = functions[0]
        self.thread_extents = self._thread_extents()
        self.cta_threads = self.thread_extents.get("threadIdx.x")
        if self.cta_threads is None or not 1 <= self.cta_threads <= 1024:
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "threadIdx.x extent must be a static value in [1, 1024]",
                {"thread_extents": self.thread_extents},
            )
        for dimension in ("threadIdx.y", "threadIdx.z"):
            if self.thread_extents.get(dimension, 1) != 1:
                raise CaptureError(
                    "TILELANG_MULTIDIMENSIONAL_THREADS_UNSUPPORTED",
                    "initial TileLang capture requires unit threadIdx.y/z extents",
                    {"thread_extents": self.thread_extents},
                )
        for dimension in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
            if self.thread_extents.get(dimension, 1) != 1:
                raise CaptureError(
                    "TILELANG_MULTI_CTA_UNSUPPORTED",
                    "PGIR-v1 TileLang capture requires one statically selected CTA",
                    {"thread_extents": self.thread_extents},
                )
        self.actor_ids = tuple(f"lane:{lane:04d}" for lane in range(self.cta_threads))
        self.resources: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, dict[str, Any]] = {}
        self.operations: list[dict[str, Any]] = []
        self.static_expansions = 0
        self.executable_nodes_seen = 0
        self.classified_nodes = 0
        self.emitting_nodes = 0
        self.elided_nodes = 0
        self.allocation_ids: dict[Any, str] = {}
        self.used_allocation_ids: set[str] = set()
        self.global_allocation_keys: set[Any] = set()
        attrs = dict(self.func.attrs) if self.func.attrs else {}
        self.noalias = bool(attrs.get("tirx.noalias", False))
        target = attrs.get("target")
        self.target_text = str(target)
        target_kind = getattr(getattr(target, "kind", None), "name", None)
        target_attrs = dict(getattr(target, "attrs", {})) if target is not None else {}
        target_arch = str(target_attrs.get("arch", ""))
        target_warp_size = int(target_attrs.get("thread_warp_size", 0))
        if target_kind != "cuda" or target_arch not in {"sm_90", "sm_90a"} or target_warp_size != 32:
            raise CaptureError(
                "TILELANG_TARGET_UNSUPPORTED",
                "TileLang capture supports only CUDA SM90 targets with 32-lane warps",
                {
                    "target": self.target_text,
                    "kind": target_kind,
                    "arch": target_arch,
                    "warp_size": target_warp_size,
                },
            )
        self.target_rule = f"tilelang-cuda-{target_arch}-v1"

    def classify_leaf(self, operation_count_before: int) -> None:
        self.executable_nodes_seen += 1
        self.classified_nodes += 1
        if len(self.operations) > operation_count_before:
            self.emitting_nodes += 1
        else:
            self.elided_nodes += 1

    def allocation_id(self, data: Any, scope: str, actor: str, label: str) -> str:
        key: Any = (data, actor) if scope == "local" else data
        if key in self.allocation_ids:
            return self.allocation_ids[key]
        if scope == "global":
            if self.global_allocation_keys and not self.noalias:
                raise CaptureError(
                    "TILELANG_ALIAS_UNRESOLVED",
                    "multiple global pointer bases require tirx.noalias evidence",
                    {"allocation": label},
                )
            self.global_allocation_keys.add(key)
        preferred = f"{actor}:{label}" if scope == "local" else label
        identifier = preferred
        suffix = 1
        while identifier in self.used_allocation_ids:
            identifier = f"{preferred}:{suffix}"
            suffix += 1
        self.allocation_ids[key] = identifier
        self.used_allocation_ids.add(identifier)
        return identifier

    def allocation_for_buffer(self, buffer: Any, actor: str) -> str:
        scope = str(buffer.scope())
        label = getattr(buffer.data, "name", None) or buffer.name
        return self.allocation_id(buffer.data, scope, actor, label)

    def _thread_extents(self) -> dict[str, int]:
        attrs = dict(self.func.attrs) if self.func.attrs else {}
        raw = attrs.get("thread_extent")
        if raw is None or not hasattr(raw, "items"):
            raise CaptureError(
                "TILELANG_PARTICIPANTS_UNRESOLVED",
                "lowered PrimFunc lacks static thread_extent metadata",
            )
        result: dict[str, int] = {}
        for key, value in raw.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError) as error:
                raise CaptureError(
                    "TILELANG_PARTICIPANTS_UNRESOLVED",
                    "thread extent is not an integer",
                    {"dimension": str(key), "value": str(value)},
                ) from error
        return result

    def capture(self) -> dict[str, Any]:
        for lane in range(self.cta_threads):
            walker = LaneWalker(self, lane)
            walker.walk(self.func.body, ("root",))
            walker.finalize()
        has_mbarrier = any(
            resource["kind"] == "mbarrier" for resource in self.resources.values()
        )
        if has_mbarrier and not any(
            operation["op"] == "tilelang.mbarrier.init"
            for operation in self.operations
        ):
            raise CaptureError(
                "TILELANG_MBARRIER_INIT_MISSING",
                "captured mbarrier operations require an explicit static initialization",
            )
        tilelang_root = Path(sys.modules["tilelang"].__file__).resolve().parents[1]
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "profile": {
                "id": "sm90-single-cta-static-v1",
                "source_ir_sha256": self.source_ir_sha256,
                "compiler_ir_stage": "post-inject-fence-proxy",
                "target_rule": self.target_rule,
            },
            "producer": {
                "dsl_version": str(getattr(sys.modules["tilelang"], "__version__", "unavailable")),
                "dsl_commit": _git_revision(tilelang_root),
                "backend_commit": str(getattr(self.tvm, "__version__", "tilelang-tirx")),
                "adapter_version": ADAPTER_VERSION,
                "command": self.command,
            },
            "target": {
                "architecture": "sm90",
                "warp_size": 32,
                "cta_threads": self.cta_threads,
                "shared_memory_bytes": 65536,
                "grid_scope": "single_cta",
            },
            "extraction": {
                "complete": True,
                "stage": "post-inject-fence-proxy",
                "source": self.source,
                "finite_control_flow": True,
                "statically_known_participants": True,
                "fair_async_completion": True,
                "no_cluster_effects": True,
                "alias_precision": "EXACT",
                "parity_normalization_status": "RESOLVED",
                "static_expansions": self.static_expansions,
                "executable_nodes_seen": self.executable_nodes_seen,
                "classified_nodes": self.classified_nodes,
                "emitting_nodes": self.emitting_nodes,
                "elided_nodes": self.elided_nodes,
                "events_emitted": len(self.operations),
                "allocation_identity_evidence": "tir-buffer-data-object-and-actor-identity-v1",
                "alias_evidence": "tirx.noalias-v1" if self.noalias else "single-global-base-v1",
                "participant_evidence": "primfunc-thread-extent-lane-enumeration-v1",
            },
            "actors": [
                {"id": actor, "kind": "LANE", "participants": [lane]}
                for lane, actor in enumerate(self.actor_ids)
            ],
            "resources": [self.resources[key] for key in sorted(self.resources)],
            "regions": [self.regions[key] for key in sorted(self.regions)],
            "operations": sorted(
                self.operations,
                key=lambda operation: (operation["actor"], operation["seq"], operation["id"]),
            ),
            "unsupported_constructs": [],
        }

    def mbarrier(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> str:
        allocation = self.allocation_for_buffer(buffer, evaluator.actor)
        offset, extent = evaluator.byte_interval(buffer, indices)
        if "barrier" not in evaluator.buffer_scope(buffer):
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier operand is not in shared.barrier storage",
                {"buffer": buffer.name, "scope": evaluator.buffer_scope(buffer)},
            )
        if extent != 8 or offset % 8:
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier storage must be one aligned eight-byte object",
                {"buffer": buffer.name, "offset": offset, "extent": extent},
            )
        identifier = f"mbarrier:{allocation}:{offset}:{extent}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "mbarrier",
            "participants": list(self.actor_ids),
        })
        return identifier

    def cta_barrier(self, path: tuple[Any, ...]) -> str:
        identifier = f"cta:{_stable_path(path)}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "cta_barrier",
            "participants": list(self.actor_ids),
            "capacity": self.cta_threads,
            "scope": "CTA",
            "visibility_mask": {
                "shared": True,
                "global_read": False,
                "global_write": False,
                "tensor_read": False,
                "tensor_write": False,
                "control_only": False,
            },
        })
        return identifier

    def async_group(self, actor: str) -> str:
        identifier = f"cp_async:{actor}"
        self.resources.setdefault(identifier, {
            "id": identifier,
            "kind": "async_group",
            "participants": [actor],
            "capacity": 8,
            "protocol": "CP_ASYNC",
            "scope": "CTA",
        })
        return identifier

    def region(self, allocation: str, scope: str, offset: int, extent: int) -> str:
        address_space = {
            "shared": "SHARED",
            "shared.dyn": "SHARED",
            "global": "GLOBAL",
            "local": "REGISTER",
        }.get(scope)
        if address_space is None:
            raise CaptureError(
                "TILELANG_ADDRESS_SPACE_UNSUPPORTED",
                "buffer scope has no sound PGIR mapping",
                {"allocation": allocation, "scope": scope},
            )
        if offset < 0 or extent <= 0:
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "memory region requires a nonnegative offset and positive extent",
                {"allocation": allocation, "offset": offset, "extent": extent},
            )
        identifier = f"region:{address_space.lower()}:{allocation}:{offset}:{extent}"
        self.regions.setdefault(identifier, {
            "id": identifier,
            "allocation_id": allocation,
            "address_space": address_space,
            "offset": offset,
            "extent": extent,
        })
        return identifier

    def memory_region(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> str:
        allocation = self.allocation_for_buffer(buffer, evaluator.actor)
        offset, extent = evaluator.byte_interval(buffer, indices)
        return self.region(allocation, evaluator.buffer_scope(buffer), offset, extent)


class LaneWalker:
    def __init__(self, capture: TileLangCapture, lane: int):
        self.capture = capture
        self.tvm = capture.tvm
        self.tirx = capture.tirx
        self.lane = lane
        self.actor = capture.actor_ids[lane]
        self.bindings: dict[Any, int] = {}
        self.sequence = 0
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}
        self.cp_group_seq = 0
        self.cp_staged = 0
        self.pending_proxy: dict[str, Any] | None = None
        self.current_ir_operation = "unknown"

    def finalize(self) -> None:
        if self.pending_proxy is not None:
            raise CaptureError(
                "TILELANG_PROXY_TARGET_UNRESOLVED",
                "fence_proxy_async has no immediately following supported async issue",
                {"event": self.pending_proxy["id"]},
            )

    def emit(self, op: str, path: tuple[Any, ...], **fields: Any) -> None:
        ir_operation = fields.pop("_ir_operation", self.current_ir_operation)
        recognizer_rule = fields.pop(
            "_recognizer_rule",
            f"{ir_operation}->{op}:v1",
        )
        if self.pending_proxy is not None and op != "tilelang.cp_async.issue":
            raise CaptureError(
                "TILELANG_PROXY_TARGET_UNRESOLVED",
                "fence_proxy_async must be followed immediately by a supported async issue",
                {"event": self.pending_proxy["id"], "next_operation": op},
            )
        identifier = f"tl:{self.actor}:e{self.sequence:05d}:{_stable_path((*path, op))}"
        self.capture.operations.append({
            "id": identifier,
            "op": op,
            "actor": self.actor,
            "seq": self.sequence,
            "evidence": {
                "source_location": "/".join(map(str, path)),
                "activation_path": [str(item) for item in path],
                "recognizer_rule": recognizer_rule,
                "ir_operation": ir_operation,
            },
            **fields,
        })
        self.sequence += 1

    def eval_int(self, expression: Any) -> int | None:
        if isinstance(expression, self.tirx.IntImm):
            return int(expression.value)
        if isinstance(expression, self.tirx.Var):
            return self.bindings.get(expression)
        try:
            replacements = {
                variable: self.tirx.IntImm(variable.dtype, value)
                for variable, value in self.bindings.items()
            }
            substituted = self.tirx.stmt_functor.substitute(expression, replacements)
            simplified = self.capture.tvm.arith.Analyzer().simplify(substituted)
            if isinstance(simplified, self.tirx.IntImm):
                return int(simplified.value)
        except Exception:
            pass
        name = type(expression).__name__
        if name == "Cast":
            return self.eval_int(expression.value)
        if name == "Not":
            value = self.eval_int(expression.a)
            return None if value is None else int(not value)
        if name == "Select":
            condition = self.eval_int(expression.condition)
            if condition is None:
                return None
            return self.eval_int(expression.true_value if condition else expression.false_value)
        if name == "Let":
            value = self.eval_int(expression.value)
            if value is None:
                return None
            old = self.bindings.get(expression.var)
            self.bindings[expression.var] = value
            result = self.eval_int(expression.body)
            if old is None:
                self.bindings.pop(expression.var, None)
            else:
                self.bindings[expression.var] = old
            return result

        def trunc_div(left: int, right: int) -> int:
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient

        binary = {
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b,
            "Div": trunc_div,
            "FloorDiv": lambda a, b: a // b,
            "Mod": lambda a, b: a - trunc_div(a, b) * b,
            "FloorMod": lambda a, b: a % b,
            "Min": min,
            "Max": max,
            "EQ": lambda a, b: int(a == b),
            "NE": lambda a, b: int(a != b),
            "LT": lambda a, b: int(a < b),
            "LE": lambda a, b: int(a <= b),
            "GT": lambda a, b: int(a > b),
            "GE": lambda a, b: int(a >= b),
            "And": lambda a, b: int(bool(a) and bool(b)),
            "Or": lambda a, b: int(bool(a) or bool(b)),
        }
        function = binary.get(name)
        if function is not None:
            left = self.eval_int(expression.a)
            right = self.eval_int(expression.b)
            if left is None or right is None or (name in {"Div", "FloorDiv", "Mod", "FloorMod"} and right == 0):
                return None
            return function(left, right)
        if isinstance(expression, self.tirx.Call):
            call_name = getattr(expression.op, "name", "")
            functions = {
                "tirx.bitwise_xor": lambda a, b: a ^ b,
                "tirx.bitwise_and": lambda a, b: a & b,
                "tirx.bitwise_or": lambda a, b: a | b,
                "tirx.shift_left": lambda a, b: a << b,
                "tirx.shift_right": lambda a, b: a >> b,
            }
            function = functions.get(call_name)
            if function is not None and len(expression.args) == 2:
                left = self.eval_int(expression.args[0])
                right = self.eval_int(expression.args[1])
                if left is not None and right is not None:
                    if call_name in {"tirx.shift_left", "tirx.shift_right"} and right < 0:
                        return None
                    return function(left, right)
        return None

    def buffer_scope(self, buffer: Any) -> str:
        raw = buffer.scope()
        return str(raw)

    def _dtype_bytes(self, buffer: Any) -> int:
        dtype = buffer.dtype
        bits = int(getattr(dtype, "bits", 0))
        lanes = int(getattr(dtype, "lanes", 1))
        if bits <= 0 or bits % 8:
            raise CaptureError(
                "TILELANG_DTYPE_UNSUPPORTED",
                "buffer dtype has no static byte width",
                {"buffer": buffer.name, "dtype": str(dtype)},
            )
        return bits // 8 * lanes

    def byte_interval(self, buffer: Any, indices: Any) -> tuple[int, int]:
        shape = [self.eval_int(item) for item in buffer.shape]
        index_values = [self.eval_int(item) for item in indices]
        elem_offset = self.eval_int(buffer.elem_offset)
        raw_strides = list(buffer.strides)
        strides = [self.eval_int(item) for item in raw_strides]
        unresolved = shape + index_values + strides + [elem_offset]
        if (
            len(shape) != len(index_values)
            or (raw_strides and len(strides) != len(shape))
            or any(value is None for value in unresolved)
        ):
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "buffer region offset, strides, shape, or indices are symbolic",
                {
                    "buffer": buffer.name,
                    "indices": [str(item) for item in indices],
                    "strides": [str(item) for item in raw_strides],
                    "elem_offset": str(buffer.elem_offset),
                },
            )
        assert elem_offset is not None
        for dimension, index in zip(shape, index_values):
            assert dimension is not None and index is not None
            if dimension <= 0 or not 0 <= index < dimension:
                raise CaptureError(
                    "TILELANG_REGION_OUT_OF_BOUNDS",
                    "buffer access is outside its static shape",
                    {"buffer": buffer.name, "shape": shape, "indices": index_values},
                )
        if raw_strides:
            linear = elem_offset + sum(
                int(index) * int(stride)
                for index, stride in zip(index_values, strides)
            )
        else:
            linear = 0
            for dimension, index in zip(shape, index_values):
                assert dimension is not None and index is not None
                linear = linear * dimension + index
            linear += elem_offset
        width = self._dtype_bytes(buffer)
        return linear * width, width

    def access_ptr_region(self, expression: Any, required_flag: int) -> tuple[str, int]:
        if (
            not isinstance(expression, self.tirx.Call)
            or getattr(expression.op, "name", "") not in {"tirx.tvm_access_ptr", "tir.tvm_access_ptr"}
            or len(expression.args) != 5
        ):
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async memory operand must be a five-argument tvm_access_ptr",
                {"expression": str(expression)},
            )
        offset = self.eval_int(expression.args[2])
        extent = self.eval_int(expression.args[3])
        flags = self.eval_int(expression.args[4])
        data = expression.args[1]
        allocation_label = getattr(data, "name", None)
        if offset is None or extent is None or flags is None or allocation_label is None:
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async access pointer has symbolic offset, extent, flags, or allocation",
                {"expression": str(expression)},
            )
        if flags & required_flag != required_flag:
            raise CaptureError(
                "TILELANG_ACCESS_MODE_MISMATCH",
                "async access pointer flags do not include the required mode",
                {"flags": flags, "required_flag": required_flag},
            )
        import re

        dtype_match = re.search(r"(?:u?int|float|bfloat)(\d+)", str(expression.args[0]))
        scope_values = re.findall(r'"([^"]+)"', str(getattr(expression.args[1], "type_annotation", "")))
        if dtype_match is None or not scope_values:
            raise CaptureError(
                "TILELANG_ACCESS_PTR_UNRESOLVED",
                "async access pointer dtype or address space is unresolved",
                {"expression": str(expression)},
            )
        bits = int(dtype_match.group(1))
        if bits <= 0 or bits % 8:
            raise CaptureError(
                "TILELANG_DTYPE_UNSUPPORTED",
                "async access pointer element width is not byte-aligned",
                {"dtype": str(expression.args[0])},
            )
        width = bits // 8
        byte_offset = offset * width
        byte_extent = extent * width
        scope = scope_values[-1]
        allocation = self.capture.allocation_id(data, scope, self.actor, allocation_label)
        region = self.capture.region(allocation, scope, byte_offset, byte_extent)
        return region, byte_extent

    def _access_enabled(self, access: Any) -> bool:
        predicate = getattr(access, "predicate", None)
        if predicate is None:
            return True
        value = self.eval_int(predicate)
        if value is None:
            raise CaptureError(
                "TILELANG_PREDICATE_UNRESOLVED",
                "memory predicate is not statically resolvable",
                {"predicate": str(predicate)},
            )
        return bool(value)

    def _buffer_loads(self, expression: Any) -> list[Any]:
        loads: list[Any] = []
        unsupported_effects: list[dict[str, Any]] = []
        deferred_errors: list[CaptureError] = []

        def visit(node: Any) -> None:
            if isinstance(node, self.tirx.BufferLoad):
                try:
                    if self._access_enabled(node):
                        loads.append(node)
                except CaptureError as error:
                    deferred_errors.append(error)
                return
            if not isinstance(node, self.tirx.Call):
                return
            effect = node.op.get_attr("TCallEffectKind")
            effect_value = getattr(effect, "value", None)
            pure_values = {
                self.tirx.CallEffectKind.ExprAnnotation.value,
                self.tirx.CallEffectKind.Pure.value,
            }
            if effect_value not in pure_values:
                unsupported_effects.append({
                    "operation": getattr(node.op, "name", ""),
                    "effect_kind": effect_value,
                })

        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression), visit)
        if deferred_errors:
            raise deferred_errors[0]
        if unsupported_effects:
            raise CaptureError(
                "TILELANG_EXPRESSION_EFFECT_UNSUPPORTED",
                "nested TIRX call is not proven side-effect free",
                {"calls": unsupported_effects},
            )
        return loads

    def _emit_expression_loads(self, expression: Any, path: tuple[Any, ...]) -> None:
        for index, load in enumerate(self._buffer_loads(expression)):
            region = self.capture.memory_region(load.buffer, load.indices, self)
            self.emit(
                "tilelang.mem.read",
                (*path, "read", index),
                _ir_operation="BufferLoad",
                region=region,
            )

    def walk(self, statement: Any, path: tuple[Any, ...]) -> None:
        if statement is None:
            return
        name = type(statement).__name__
        if name == "SeqStmt":
            for index, child in enumerate(statement.seq):
                self.walk(child, (*path, "seq", index))
            return
        if name == "AttrStmt":
            if statement.attr_key == "thread_extent":
                tag = str(getattr(statement.node, "thread_tag", ""))
                extent = self.eval_int(statement.value)
                variable = getattr(statement.node, "var", None)
                if extent is None or variable is None:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "thread binding is not statically resolvable",
                        {"tag": tag, "path": list(path)},
                    )
                had_old = variable in self.bindings
                old = self.bindings.get(variable)
                if tag == "threadIdx.x":
                    self.bindings[variable] = self.lane
                elif tag in {"threadIdx.y", "threadIdx.z", "blockIdx.x", "blockIdx.y", "blockIdx.z"}:
                    if extent != 1:
                        raise CaptureError(
                            "TILELANG_SCOPE_UNSUPPORTED",
                            "non-x thread and block bindings must have unit extent",
                            {"tag": tag, "extent": extent},
                        )
                    self.bindings[variable] = 0
                else:
                    raise CaptureError(
                        "TILELANG_THREAD_BINDING_UNSUPPORTED",
                        "thread binding tag is unsupported",
                        {"tag": tag},
                    )
                self.walk(statement.body, (*path, "attr", tag))
                if had_old:
                    assert old is not None
                    self.bindings[variable] = old
                else:
                    self.bindings.pop(variable, None)
                return
            transparent_attributes = {
                "buffer_dim_align",
                "double_buffer_scope",
                "pipeline_exec_scope",
                "pipeline_scope",
                "pragma_auto_unroll_max_step",
                "pragma_unroll_explicit",
                "realize_scope",
                "software_pipeline_order",
                "software_pipeline_stage",
                "storage_scope",
            }
            if statement.attr_key in transparent_attributes:
                self.walk(statement.body, (*path, "attr", statement.attr_key))
                return
            raise CaptureError(
                "TILELANG_ATTRIBUTE_UNSUPPORTED",
                "lowered TileLang statement attribute is unsupported",
                {"attribute": statement.attr_key, "path": list(path)},
            )
        if name in {"AllocBuffer", "DeclBuffer"}:
            self.capture.allocation_for_buffer(statement.buffer, self.actor)
            body = getattr(statement, "body", None)
            if body is not None:
                self.walk(body, (*path, "buffer", statement.buffer.name))
            return
        if name == "Bind":
            before = len(self.capture.operations)
            value = self.eval_int(statement.value)
            if value is None:
                raise CaptureError(
                    "TILELANG_STATIC_VALUE_UNRESOLVED",
                    "TIRX Bind value is not statically resolvable",
                    {"variable": str(statement.var), "value": str(statement.value)},
                )
            self.bindings[statement.var] = value
            self.capture.classify_leaf(before)
            return
        if name == "LetStmt":
            value = self.eval_int(statement.value)
            if value is None:
                raise CaptureError(
                    "TILELANG_STATIC_VALUE_UNRESOLVED",
                    "TIRX LetStmt value is not statically resolvable",
                    {"variable": str(statement.var), "value": str(statement.value)},
                )
            had_old = statement.var in self.bindings
            old = self.bindings.get(statement.var)
            self.bindings[statement.var] = value
            self.walk(statement.body, (*path, "let", statement.var.name))
            if had_old:
                assert old is not None
                self.bindings[statement.var] = old
            else:
                self.bindings.pop(statement.var, None)
            return
        if name == "AssertStmt":
            condition = self.eval_int(statement.condition)
            if condition is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TIRX assertion condition is not statically resolvable",
                    {"condition": str(statement.condition)},
                )
            if not condition:
                raise CaptureError(
                    "TILELANG_STATIC_ASSERTION_FAILED",
                    "TIRX static assertion is false",
                    {"condition": str(statement.condition)},
                )
            self.walk(statement.body, (*path, "assert"))
            return
        if name == "IfThenElse":
            condition = self.eval_int(statement.condition)
            if condition is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TileLang branch condition cannot be evaluated per lane",
                    {"condition": str(statement.condition), "lane": self.lane},
                )
            branch = statement.then_case if condition else statement.else_case
            self.walk(branch, (*path, "if", int(bool(condition))))
            return
        if name == "While":
            seen_states: set[frozenset[tuple[Any, int]]] = set()
            iteration = 0
            while True:
                condition = self.eval_int(statement.condition)
                if condition is None:
                    raise CaptureError(
                        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                        "TileLang while condition cannot be evaluated statically",
                        {"condition": str(statement.condition), "iteration": iteration},
                    )
                if not condition:
                    break
                state = frozenset(self.bindings.items())
                if state in seen_states:
                    raise CaptureError(
                        "TILELANG_NONFINITE_CONTROL_FLOW",
                        "TileLang while loop repeats a static state",
                        {"condition": str(statement.condition), "iteration": iteration},
                    )
                seen_states.add(state)
                self.capture.static_expansions += 1
                self.walk(statement.body, (*path, "while", iteration))
                iteration += 1
            return
        if name == "For":
            minimum = self.eval_int(statement.min)
            extent = self.eval_int(statement.extent)
            if minimum is None or extent is None or extent < 0:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TileLang loop must have a finite static extent",
                    {"loop": str(statement.loop_var), "extent": str(statement.extent)},
                )
            variable = statement.loop_var
            had_old = variable in self.bindings
            old = self.bindings.get(variable)
            for iteration in range(minimum, minimum + extent):
                self.capture.static_expansions += 1
                self.bindings[variable] = iteration
                self.walk(statement.body, (*path, "for", variable.name, iteration))
            if had_old:
                assert old is not None
                self.bindings[variable] = old
            else:
                self.bindings.pop(variable, None)
            return
        if name == "BufferStore":
            before = len(self.capture.operations)
            self.current_ir_operation = name
            for index, load in enumerate(self._buffer_loads(statement.value)):
                region = self.capture.memory_region(load.buffer, load.indices, self)
                self.emit(
                    "tilelang.mem.read",
                    (*path, "rhs", index),
                    _ir_operation="BufferLoad",
                    region=region,
                )
            if self._access_enabled(statement):
                region = self.capture.memory_region(statement.buffer, statement.indices, self)
                self.emit("tilelang.mem.write", path, region=region)
            self.capture.classify_leaf(before)
            return
        if name == "Evaluate":
            before = len(self.capture.operations)
            self.evaluate(statement.value, path)
            self.capture.classify_leaf(before)
            return
        if name in {"BlockRealize", "SBlockRealize"}:
            block = statement.block
            iter_vars = list(block.iter_vars)
            iter_values = list(statement.iter_values)
            if len(iter_vars) != len(iter_values):
                raise CaptureError(
                    "TILELANG_BLOCK_BINDING_UNRESOLVED",
                    "BlockRealize iter values do not match block iter variables",
                    {"block": str(block.name_hint)},
                )
            saved: list[tuple[Any, bool, int | None]] = []
            for iter_var, value_expression in zip(iter_vars, iter_values):
                value = self.eval_int(value_expression)
                if value is None:
                    raise CaptureError(
                        "TILELANG_BLOCK_BINDING_UNRESOLVED",
                        "BlockRealize iter value is not statically resolvable",
                        {"block": str(block.name_hint), "value": str(value_expression)},
                    )
                saved.append((iter_var.var, iter_var.var in self.bindings, self.bindings.get(iter_var.var)))
                self.bindings[iter_var.var] = value
            predicate = self.eval_int(statement.predicate)
            if predicate is None:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "BlockRealize predicate is not statically resolvable",
                    {"block": str(block.name_hint), "predicate": str(statement.predicate)},
                )
            if predicate:
                reduction_vars = [
                    iter_var for iter_var in iter_vars
                    if int(iter_var.iter_type) == self.tirx.IterVar.CommReduce
                ]
                run_init = block.init is not None
                for iter_var in reduction_vars:
                    minimum = self.eval_int(iter_var.dom.min) if iter_var.dom is not None else None
                    if minimum is None:
                        raise CaptureError(
                            "TILELANG_BLOCK_INIT_UNRESOLVED",
                            "reduction block minimum is not statically resolvable",
                            {"block": str(block.name_hint)},
                        )
                    run_init = run_init and self.bindings[iter_var.var] == minimum
                if run_init:
                    self.walk(block.init, (*path, "block", block.name_hint, "init"))
                self.walk(block.body, (*path, "block", block.name_hint, "body"))
            for variable, had_old, old in reversed(saved):
                if had_old:
                    assert old is not None
                    self.bindings[variable] = old
                else:
                    self.bindings.pop(variable, None)
            return
        if name in {"Block", "SBlock"}:
            if statement.init is not None:
                self.walk(statement.init, (*path, "block", statement.name_hint, "init"))
            self.walk(statement.body, (*path, "block", statement.name_hint, "body"))
            return
        raise CaptureError(
            "TILELANG_STATEMENT_UNSUPPORTED",
            "lowered TileLang statement has no sound capture rule",
            {"statement": name, "path": list(path)},
        )

    def evaluate(self, expression: Any, path: tuple[Any, ...]) -> None:
        self.current_ir_operation = type(expression).__name__
        if isinstance(expression, self.tirx.IntImm):
            return
        if isinstance(expression, self.tirx.BufferLoad):
            if self._access_enabled(expression):
                region = self.capture.memory_region(expression.buffer, expression.indices, self)
                self.emit("tilelang.mem.read", path, region=region)
            return
        if not isinstance(expression, self.tirx.Call):
            self._emit_expression_loads(expression, path)
            return
        name = getattr(expression.op, "name", "")
        self.current_ir_operation = name
        if name == "tl.ptx_fence_barrier_init":
            if expression.args:
                raise CaptureError(
                    "TILELANG_OPERATION_MALFORMED",
                    "ptx_fence_barrier_init must not have operands",
                )
            return
        if name == "tl.fence_proxy_async":
            if expression.args or self.pending_proxy is not None:
                raise CaptureError(
                    "TILELANG_PROXY_TARGET_UNRESOLVED",
                    "fence_proxy_async must be operand-free and have one unresolved target",
                )
            self.emit(
                "tilelang.proxy_fence",
                path,
                from_domain="GENERIC",
                to_domain="ASYNC_COPY",
                scope="CTA",
                regions=[],
                visibility_mask={
                    "shared": False,
                    "global_read": False,
                    "global_write": False,
                    "tensor_read": False,
                    "tensor_write": False,
                    "control_only": False,
                },
            )
            self.pending_proxy = self.capture.operations[-1]
            return
        if name in {"tirx.tvm_storage_sync", "tir.tvm_storage_sync"}:
            if len(expression.args) != 1 or str(expression.args[0]).strip('"') != "shared":
                raise CaptureError(
                    "TILELANG_BARRIER_SCOPE_UNSUPPORTED",
                    "only full-CTA shared storage synchronization is supported",
                    {"operation": name, "args": [str(arg) for arg in expression.args]},
                )
            resource = self.capture.cta_barrier(path)
            self.emit(
                "tilelang.barrier.arrive",
                (*path, "arrive"),
                resource=resource,
                invocation_id=0,
                count=1,
            )
            self.emit(
                "tilelang.barrier.observe",
                (*path, "observe"),
                resource=resource,
                invocation_id=0,
            )
            return
        if name in {"tirx.ptx_init_barrier_thread_count", "tir.ptx_init_barrier_thread_count"}:
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_MBARRIER_COUNT_UNRESOLVED",
                    "mbarrier initialization requires barrier and count operands",
                    {"operation": name, "operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            count = self.eval_int(expression.args[1])
            if count is None or count <= 0:
                raise CaptureError(
                    "TILELANG_MBARRIER_COUNT_UNRESOLVED",
                    "mbarrier initialization count must be a static positive integer",
                    {"operation": name},
                )
            self.emit(
                "tilelang.mbarrier.init",
                path,
                resource=resource,
                lifecycle_epoch=0,
                count=count,
            )
            return
        if name in {"tirx.ptx_arrive_barrier", "tir.ptx_arrive_barrier"}:
            if len(expression.args) != 1:
                raise CaptureError(
                    "TILELANG_MBARRIER_ARRIVE_UNRESOLVED",
                    "mbarrier arrival requires exactly one barrier operand",
                    {"operation": name, "operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            generation = self.arrival_rank.get(resource, 0)
            self.arrival_rank[resource] = generation + 1
            self.emit(
                "tilelang.mbarrier.arrive",
                path,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                count=1,
                predicate=True,
            )
            return
        if name == "tl.mbarrier_wait_parity":
            if len(expression.args) != 2:
                raise CaptureError(
                    "TILELANG_PARITY_UNRESOLVED",
                    "mbarrier wait requires barrier and parity operands",
                    {"operands": len(expression.args)},
                )
            resource = self._barrier_operand(expression, 0)
            parity = self.eval_int(expression.args[1])
            generation = self.wait_rank.get(resource, 0)
            if parity is None or parity != generation % 2:
                raise CaptureError(
                    "TILELANG_PARITY_UNRESOLVED",
                    "wait parity does not match the recovered gap-free stream rank",
                    {"resource": resource, "rank": generation, "parity": parity},
                )
            self.wait_rank[resource] = generation + 1
            self.emit(
                "tilelang.mbarrier.wait_parity",
                path,
                resource=resource,
                lifecycle_epoch=0,
                generation=generation,
                raw_parity=parity,
                logical_stream_id=f"{self.actor}:{resource}",
                stream_rank=generation,
                polling_coalesced=True,
            )
            return
        if name == "tl.ptx_cp_async":
            if len(expression.args) != 3:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async requires destination, source, and byte count operands",
                )
            destination, destination_bytes = self.access_ptr_region(expression.args[0], 2)
            source, source_bytes = self.access_ptr_region(expression.args[1], 1)
            byte_count = self.eval_int(expression.args[2])
            if (
                byte_count is None
                or byte_count <= 0
                or byte_count > destination_bytes
                or byte_count > source_bytes
            ):
                raise CaptureError(
                    "TILELANG_CP_ASYNC_UNRESOLVED",
                    "ptx_cp_async byte count exceeds a resolved access interval",
                    {
                        "bytes": byte_count,
                        "source_extent": source_bytes,
                        "destination_extent": destination_bytes,
                    },
                )
            if self.pending_proxy is not None:
                source_space = self.capture.regions[source]["address_space"]
                mask_field = {
                    "SHARED": "shared",
                    "GLOBAL": "global_write",
                    "TENSOR": "tensor_write",
                }.get(source_space)
                if mask_field is None:
                    raise CaptureError(
                        "TILELANG_PROXY_REGION_UNSUPPORTED",
                        "fence_proxy_async source region has no supported visibility mask",
                        {"region": source, "address_space": source_space},
                    )
                self.pending_proxy["regions"] = [source]
                self.pending_proxy["visibility_mask"][mask_field] = True
                self.pending_proxy = None
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.issue",
                path,
                resource=resource,
                source_region=source,
                destination_region=destination,
                bytes=byte_count,
                group_owner=self.actor,
                predicate=True,
            )
            self.cp_staged += 1
            return
        if name in {"tirx.ptx_commit_group", "tir.ptx_commit_group"}:
            if expression.args:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_COMMIT_UNRESOLVED",
                    "ptx_commit_group must not have operands",
                )
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.commit_group",
                path,
                resource=resource,
                group_owner=self.actor,
                group_seq=self.cp_group_seq,
            )
            self.cp_group_seq += 1
            self.cp_staged = 0
            return
        if name in {"tirx.ptx_wait_group", "tir.ptx_wait_group"}:
            wait_count = self.eval_int(expression.args[0]) if len(expression.args) == 1 else None
            if wait_count is None or not 0 <= wait_count <= 7:
                raise CaptureError(
                    "TILELANG_CP_ASYNC_WAIT_UNRESOLVED",
                    "ptx_wait_group count must be a static integer in [0, 7]",
                    {"wait_group_count": wait_count},
                )
            resource = self.capture.async_group(self.actor)
            self.emit(
                "tilelang.cp_async.wait_group",
                path,
                resource=resource,
                group_owner=self.actor,
                wait_group_count=wait_count,
            )
            return
        self._emit_expression_loads(expression, path)

    def _barrier_operand(self, call: Any, index: int) -> str:
        if len(call.args) <= index or not isinstance(call.args[index], self.tirx.BufferLoad):
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier operand must be a static BufferLoad",
                {"operation": getattr(call.op, "name", "")},
            )
        operand = call.args[index]
        return self.capture.mbarrier(operand.buffer, operand.indices, self)


def capture_file(path: Path, source: str, command: str) -> dict[str, Any]:
    try:
        import tilelang
        from tilelang import tvm
        from tvm import tirx
    except ImportError as error:
        raise CaptureError(
            "TILELANG_RUNTIME_UNAVAILABLE",
            "TileLang capture must run in an isolated TileLang environment",
            {"error": str(error)},
        ) from error
    try:
        source_ir = path.read_text()
        module = tvm.ir.load_json(source_ir)
    except (OSError, ValueError, TypeError) as error:
        raise CaptureError(
            "TILELANG_IR_INPUT_INVALID",
            "TileLang IR JSON could not be loaded",
            {"path": str(path), "error": str(error)},
        ) from error
    return TileLangCapture(
        tvm,
        tirx,
        module,
        source,
        command,
        hashlib.sha256(source_ir.encode("utf-8")).hexdigest(),
    ).capture()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a strict TileLang TIRX snapshot for PipeGuard")
    parser.add_argument("input", type=Path, help="TVM JSON containing one lowered TileLang device PrimFunc")
    parser.add_argument("--source", required=True, help="stable source/provenance label")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_file(args.input, args.source, " ".join(sys.argv))
    except CaptureError as error:
        result = {
            "captured": False,
            "verdict": "UNSUPPORTED",
            "diagnostics": [error.diagnostic()],
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
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
