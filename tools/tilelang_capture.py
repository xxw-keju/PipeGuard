#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "tilelang-snapshot-v1"
ADAPTER_VERSION = "tilelang-pgir-v1"
MAX_STATIC_UNROLL = 256


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
    def __init__(self, tvm: Any, tirx: Any, module: Any, source: str, command: str):
        self.tvm = tvm
        self.tirx = tirx
        self.module = module
        self.source = source
        self.command = command
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
            LaneWalker(self, lane).walk(self.func.body, ("root",))
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
        target = dict(self.func.attrs).get("target") if self.func.attrs else None
        target_text = str(target)
        if "sm_90" not in target_text:
            raise CaptureError(
                "TILELANG_TARGET_UNSUPPORTED",
                "TileLang capture supports only SM90 targets",
                {"target": target_text},
            )
        return {
            "snapshot_version": SNAPSHOT_VERSION,
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
        allocation = getattr(buffer.data, "name", None) or buffer.name
        offset, extent = evaluator.byte_interval(buffer, indices)
        if "barrier" not in evaluator.buffer_scope(buffer):
            raise CaptureError(
                "TILELANG_BARRIER_ADDRESS_UNRESOLVED",
                "mbarrier operand is not in shared.barrier storage",
                {"buffer": buffer.name, "scope": evaluator.buffer_scope(buffer)},
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

    def memory_region(self, buffer: Any, indices: Any, evaluator: "LaneWalker") -> str:
        scope = evaluator.buffer_scope(buffer)
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
                {"buffer": buffer.name, "scope": scope},
            )
        allocation = getattr(buffer.data, "name", None) or buffer.name
        offset, extent = evaluator.byte_interval(buffer, indices)
        identifier = f"region:{address_space.lower()}:{allocation}:{offset}:{extent}"
        self.regions.setdefault(identifier, {
            "id": identifier,
            "allocation_id": allocation,
            "address_space": address_space,
            "offset": offset,
            "extent": extent,
        })
        return identifier


class LaneWalker:
    def __init__(self, capture: TileLangCapture, lane: int):
        self.capture = capture
        self.tvm = capture.tvm
        self.tirx = capture.tirx
        self.lane = lane
        self.actor = capture.actor_ids[lane]
        self.bindings: dict[str, int] = {}
        self.sequence = 0
        self.arrival_rank: dict[str, int] = {}
        self.wait_rank: dict[str, int] = {}

    def emit(self, op: str, path: tuple[Any, ...], **fields: Any) -> None:
        identifier = f"tl:{self.actor}:e{self.sequence:05d}:{_stable_path((*path, op))}"
        self.capture.operations.append({
            "id": identifier,
            "op": op,
            "actor": self.actor,
            "seq": self.sequence,
            **fields,
        })
        self.sequence += 1

    def eval_int(self, expression: Any) -> int | None:
        if isinstance(expression, self.tirx.IntImm):
            return int(expression.value)
        if isinstance(expression, self.tirx.Var):
            return self.bindings.get(expression.name)
        unary = {"Cast": "value"}
        if type(expression).__name__ in unary:
            return self.eval_int(getattr(expression, unary[type(expression).__name__]))
        binary = {
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b,
            "Div": lambda a, b: a // b,
            "FloorDiv": lambda a, b: a // b,
            "Mod": lambda a, b: a % b,
            "FloorMod": lambda a, b: a % b,
            "EQ": lambda a, b: int(a == b),
            "NE": lambda a, b: int(a != b),
            "LT": lambda a, b: int(a < b),
            "LE": lambda a, b: int(a <= b),
            "GT": lambda a, b: int(a > b),
            "GE": lambda a, b: int(a >= b),
            "And": lambda a, b: int(bool(a) and bool(b)),
            "Or": lambda a, b: int(bool(a) or bool(b)),
        }
        function = binary.get(type(expression).__name__)
        if function is not None:
            left = self.eval_int(expression.a)
            right = self.eval_int(expression.b)
            if left is not None and right is not None:
                return function(left, right)
            return None
        if isinstance(expression, self.tirx.Call):
            name = getattr(expression.op, "name", "")
            functions = {
                "tirx.bitwise_xor": lambda a, b: a ^ b,
                "tirx.bitwise_and": lambda a, b: a & b,
                "tirx.bitwise_or": lambda a, b: a | b,
                "tirx.shift_left": lambda a, b: a << b,
                "tirx.shift_right": lambda a, b: a >> b,
            }
            function = functions.get(name)
            if function is not None and len(expression.args) == 2:
                left = self.eval_int(expression.args[0])
                right = self.eval_int(expression.args[1])
                if left is not None and right is not None:
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
        if not shape or len(shape) != len(index_values) or any(value is None for value in shape + index_values):
            raise CaptureError(
                "TILELANG_REGION_UNRESOLVED",
                "buffer region offset or shape is symbolic",
                {"buffer": buffer.name, "indices": [str(item) for item in indices]},
            )
        linear = 0
        for dimension, index in zip(shape, index_values):
            assert dimension is not None and index is not None
            if dimension <= 0 or not 0 <= index < dimension:
                raise CaptureError(
                    "TILELANG_REGION_OUT_OF_BOUNDS",
                    "buffer access is outside its static shape",
                    {"buffer": buffer.name, "shape": shape, "indices": index_values},
                )
            linear = linear * dimension + index
        width = self._dtype_bytes(buffer)
        return linear * width, width

    def _buffer_loads(self, expression: Any) -> list[Any]:
        loads: list[Any] = []

        def visit(node: Any) -> None:
            if isinstance(node, self.tirx.BufferLoad):
                loads.append(node)

        self.tirx.stmt_functor.post_order_visit(self.tirx.Evaluate(expression), visit)
        return loads

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
                variable = getattr(getattr(statement.node, "var", None), "name", None)
                if extent is None or variable is None:
                    raise CaptureError(
                        "TILELANG_PARTICIPANTS_UNRESOLVED",
                        "thread binding is not statically resolvable",
                        {"tag": tag, "path": list(path)},
                    )
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
                if old is None:
                    self.bindings.pop(variable, None)
                else:
                    self.bindings[variable] = old
                return
            raise CaptureError(
                "TILELANG_ATTRIBUTE_UNSUPPORTED",
                "lowered TileLang statement attribute is unsupported",
                {"attribute": statement.attr_key, "path": list(path)},
            )
        if name in {"AllocBuffer", "DeclBuffer"}:
            body = getattr(statement, "body", None)
            if body is not None:
                self.walk(body, (*path, "buffer", statement.buffer.name))
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
        if name == "For":
            minimum = self.eval_int(statement.min)
            extent = self.eval_int(statement.extent)
            if minimum is None or extent is None or extent < 0 or extent > MAX_STATIC_UNROLL:
                raise CaptureError(
                    "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
                    "TileLang loop must have a bounded static extent",
                    {"loop": str(statement.loop_var), "extent": str(statement.extent)},
                )
            variable = statement.loop_var.name
            old = self.bindings.get(variable)
            for iteration in range(minimum, minimum + extent):
                self.bindings[variable] = iteration
                self.walk(statement.body, (*path, "for", variable, iteration))
            if old is None:
                self.bindings.pop(variable, None)
            else:
                self.bindings[variable] = old
            return
        if name == "BufferStore":
            for index, load in enumerate(self._buffer_loads(statement.value)):
                region = self.capture.memory_region(load.buffer, load.indices, self)
                self.emit("tilelang.mem.read", (*path, "rhs", index), region=region)
            region = self.capture.memory_region(statement.buffer, statement.indices, self)
            self.emit("tilelang.mem.write", path, region=region)
            return
        if name == "Evaluate":
            self.evaluate(statement.value, path)
            return
        if name in {"Block", "BlockRealize"}:
            child = statement.body if name == "Block" else statement.block
            self.walk(child, (*path, name.lower()))
            return
        raise CaptureError(
            "TILELANG_STATEMENT_UNSUPPORTED",
            "lowered TileLang statement has no sound capture rule",
            {"statement": name, "path": list(path)},
        )

    def evaluate(self, expression: Any, path: tuple[Any, ...]) -> None:
        if isinstance(expression, self.tirx.IntImm):
            return
        if isinstance(expression, self.tirx.BufferLoad):
            region = self.capture.memory_region(expression.buffer, expression.indices, self)
            self.emit("tilelang.mem.read", path, region=region)
            return
        if not isinstance(expression, self.tirx.Call):
            raise CaptureError(
                "TILELANG_EXPRESSION_UNSUPPORTED",
                "Evaluate expression has no sound capture rule",
                {"expression": type(expression).__name__, "path": list(path)},
            )
        name = getattr(expression.op, "name", "")
        if name == "tl.ptx_fence_barrier_init":
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
            resource = self._barrier_operand(expression, 0)
            count = self.eval_int(expression.args[1]) if len(expression.args) > 1 else None
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
            resource = self._barrier_operand(expression, 0)
            parity = self.eval_int(expression.args[1]) if len(expression.args) > 1 else None
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
        raise CaptureError(
            "TILELANG_OPERATION_UNSUPPORTED",
            "lowered TileLang call has no sound capture rule",
            {"operation": name, "path": list(path)},
        )

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
        module = tvm.ir.load_json(path.read_text())
    except (OSError, ValueError, TypeError) as error:
        raise CaptureError(
            "TILELANG_IR_INPUT_INVALID",
            "TileLang IR JSON could not be loaded",
            {"path": str(path), "error": str(error)},
        ) from error
    return TileLangCapture(tvm, tirx, module, source, command).capture()


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
