#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from upstream_common import RUN_RESULT_VERSION, atomic_write, canonical_json, pretty_json, sha256_bytes, sha256_text


TERMINAL_STATUSES = {
    "CAPTURED",
    "UNSUPPORTED",
    "PROFILE_SKIPPED",
    "IMPORT_ERROR",
    "LOWERING_ERROR",
    "TIMEOUT",
    "INFRA_ERROR",
}

PROFILE_DIAGNOSTICS = {
    "TILELANG_MULTI_CTA_UNSUPPORTED",
    "TILELANG_TARGET_UNSUPPORTED",
    "TILELANG_THREAD_GEOMETRY_UNSUPPORTED",
    "TRITON_MULTI_CTA_UNSUPPORTED",
    "TRITON_TARGET_UNSUPPORTED",
    "TRITON_THREAD_GEOMETRY_UNSUPPORTED",
}


def _load_module(path: Path, name: str, import_roots: list[Path]):
    original_path = list(sys.path)
    try:
        for root in reversed(import_roots):
            value = str(root)
            if value not in sys.path:
                sys.path.insert(0, value)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def _resolve_qualname(module: Any, qualname: str) -> Any:
    value = module
    for component in qualname.split("."):
        if component == "<locals>":
            raise LookupError("local qualified names require reference replay")
        value = getattr(value, component)
    return value


def _write_json(path: Path, value: Any) -> str:
    text = pretty_json(value)
    atomic_write(path, text)
    return sha256_text(text)


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _base_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_RESULT_VERSION,
        "fingerprint": request["fingerprint"],
        "dsl": request["dsl"],
        "candidate_id": request["manifest"]["candidate_id"],
        "recipe_id": request["recipe"]["recipe_id"],
        "status": "INFRA_ERROR",
        "phase": "initialization",
        "diagnostics": [],
        "artifacts": {},
        "timings": {},
    }


def _tilelang_target(tvm: Any, recipe: dict[str, Any]):
    target = recipe["target"]
    return tvm.target.Target({
        "kind": "cuda",
        "arch": target.get("arch", "sm_90a"),
        "thread_warp_size": target.get("warp_size", 32),
        "max_num_threads": 1024,
    })


def _materialize_tilelang(module: Any, recipe: dict[str, Any], *, resolution: str) -> Any:
    materialization = recipe.get("materialization") or {}
    if resolution == "original":
        obj = _resolve_qualname(module, recipe["manifest_qualname"])
    else:
        resolver = getattr(module, "pipeguard_resolve")
        obj = resolver(**materialization.get("resolver_kwargs", {}))
    args = materialization.get("args", [])
    kwargs = materialization.get("kwargs", {})
    mode = materialization.get("mode", "auto")
    if mode == "direct":
        return obj
    if hasattr(obj, "jit_impl") and hasattr(obj.jit_impl, "get_tir"):
        return obj.jit_impl.get_tir(*args, **kwargs)
    if hasattr(obj, "get_tir"):
        return obj.get_tir(*args, **kwargs)
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def _split_tilelang_device_module(tvm: Any, module: Any) -> list[tuple[str, Any]]:
    functions: list[tuple[str, Any]] = []
    for global_var, function in module.functions.items():
        name = getattr(global_var, "name_hint", str(global_var))
        functions.append((name, tvm.IRModule({name: function})))
    return sorted(functions, key=lambda item: item[0])


def _lower_tilelang(module: Any, recipe: dict[str, Any], *, resolution: str) -> tuple[Any, list[tuple[str, Any]]]:
    import tilelang
    from tilelang import tvm

    primfunc = _materialize_tilelang(module, recipe, resolution=resolution)
    compiler = recipe.get("compiler") or {}
    target = _tilelang_target(tvm, recipe)
    target_host = compiler.get("target_host")
    pass_configs = compiler.get("pass_configs", {})
    with target, tvm.transform.PassContext(opt_level=3, config=pass_configs):
        artifact = tilelang.lower(
            primfunc,
            target=target,
            target_host=target_host,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return tvm, _split_tilelang_device_module(tvm, artifact.device_mod)


def _normalize_triton_ir(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*#loc(?:\d+)?\s*=\s*loc", line):
            continue
        line = re.sub(r"\s+loc\([^\n]*\)\s*$", "", line)
        line = re.sub(r"\s+loc\(#loc\d+\)", "", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def _materialize_triton(module: Any, recipe: dict[str, Any], *, resolution: str) -> Any:
    if resolution == "original":
        value = _resolve_qualname(module, recipe["manifest_qualname"])
    else:
        materialization = recipe.get("materialization") or {}
        value = module.pipeguard_resolve(**materialization.get("resolver_kwargs", {}))
    while not hasattr(value, "src") and hasattr(value, "fn"):
        value = value.fn
    return value


def _lower_triton(module: Any, recipe: dict[str, Any], *, resolution: str) -> str:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    function = _materialize_triton(module, recipe, resolution=resolution)
    compiler = recipe.get("compiler") or {}
    source = ASTSource(
        fn=function,
        signature=compiler["signature"],
        constexprs=compiler.get("constexprs", {}),
        attrs=compiler.get("attrs", {}),
    )
    target = GPUTarget("cuda", 90, recipe["target"].get("warp_size", 32))
    compiled = triton.compile(source, target=target, options=compiler.get("options", {}))
    return str(compiled.asm["ttgir"])


def _direct_reference_module(request: dict[str, Any], name: str):
    source_root = Path(request["source_root"]).resolve(strict=True)
    manifest = request["manifest"]
    if manifest["upstream"]["qualname"].count("."):
        raise LookupError("direct original replay only supports module-level candidates")
    source_path = source_root / manifest["upstream"]["relative_path"]
    return _load_module(source_path, name, [source_path.parent, source_root])


def _reference_replay_module(request: dict[str, Any], name: str, output: Path):
    import ast
    from upstream_extract import AliasTable, CandidateVisitor, dependency_nodes, make_capsule

    source_root = Path(request["source_root"]).resolve(strict=True)
    manifest = request["manifest"]
    source_path = source_root / manifest["upstream"]["relative_path"]
    source = source_path.read_text(encoding="utf-8")
    if sha256_text(source) != manifest["upstream"]["source_sha256"]:
        raise RuntimeError("locked upstream source hash mismatch")
    tree = ast.parse(source, filename=manifest["upstream"]["relative_path"])
    visitor = CandidateVisitor(
        request["dsl"], manifest["upstream"]["relative_path"], source, AliasTable(tree)
    )
    visitor.visit(tree)
    candidate = next(
        (item for item in visitor.candidates if item.candidate_id == manifest["candidate_id"]),
        None,
    )
    if candidate is None:
        raise RuntimeError("candidate is not reproducible from locked upstream source")
    dependencies, unresolved = dependency_nodes(candidate, tree)
    capsule = make_capsule(candidate, dependencies, unresolved, source.splitlines(keepends=True))
    replay_path = output / "reference-replay.py"
    atomic_write(replay_path, capsule)
    tracked_root = Path(request["corpus_root"]).resolve(strict=True)
    tracked_path = tracked_root / manifest["capsule"]["path"]
    tracked = tracked_path.read_text(encoding="utf-8")
    if sha256_text(tracked) != manifest["capsule"]["sha256"]:
        raise RuntimeError("tracked capsule hash mismatch")
    if sha256_text(capsule) != manifest["capsule"]["sha256"]:
        raise RuntimeError("reference replay does not reproduce tracked capsule")
    return _load_module(replay_path, name, [replay_path.parent, source_root])


def _capsule_module(request: dict[str, Any], name: str):
    corpus_root = Path(request["corpus_root"]).resolve(strict=True)
    capsule_path = corpus_root / request["manifest"]["capsule"]["path"]
    return _load_module(capsule_path, name, [capsule_path.parent, Path(request["source_root"])])


def _capture_tilelang(request: dict[str, Any], raw_path: Path, output: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    import tilelang_capture

    compiler = request["recipe"].get("compiler") or {}
    evidence = compiler.get("descriptor_evidence")
    evidence_path = None
    if evidence is not None:
        evidence = dict(evidence)
        evidence["source_ir_sha256"] = sha256_text(raw_path.read_text())
        evidence_path = output / "descriptor-evidence.json"
        atomic_write(evidence_path, pretty_json(evidence))
    try:
        snapshot = tilelang_capture.capture_file(
            raw_path,
            source=f"upstream:{request['manifest']['upstream']['relative_path']}:{request['manifest']['upstream']['qualname']}",
            command=f"pipeguard-upstream:{request['fingerprint']}",
            descriptor_evidence_path=evidence_path,
        )
        return snapshot, None
    except tilelang_capture.CaptureError as error:
        return None, error.diagnostic()


def _capture_triton(request: dict[str, Any], raw_path: Path, output: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    import triton_capture

    compiler = request["recipe"].get("compiler") or {}
    evidence = compiler.get("binding_evidence")
    evidence_path = None
    if evidence is not None:
        evidence = dict(evidence)
        evidence["source_ir_sha256"] = sha256_bytes(raw_path.read_bytes())
        evidence_path = output / "binding-evidence.json"
        atomic_write(evidence_path, pretty_json(evidence))
    try:
        snapshot = triton_capture.capture_file(
            raw_path,
            source=f"upstream:{request['manifest']['upstream']['relative_path']}:{request['manifest']['upstream']['qualname']}",
            command=f"pipeguard-upstream:{request['fingerprint']}",
            grid_ctas=compiler["grid_ctas"],
            binding_evidence_path=evidence_path,
        )
        return snapshot, None
    except triton_capture.CaptureError as error:
        return None, error.diagnostic()


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result = _base_result(request)
    recipe = dict(request["recipe"])
    recipe["manifest_qualname"] = request["manifest"]["upstream"]["qualname"]
    request = dict(request)
    request["recipe"] = recipe
    if not recipe.get("lowerable"):
        result["status"] = "UNSUPPORTED"
        result["phase"] = "recipe"
        result["diagnostics"] = [_diagnostic(
            "UPSTREAM_RECIPE_UNRESOLVED",
            "candidate has no complete concrete specialization recipe",
            unresolved=recipe.get("unresolved", []),
        )]
        return result
    reference_mode = (recipe.get("reference") or {}).get("mode", "not_run")
    if reference_mode not in {"direct_original", "reference_replay"}:
        result["status"] = "UNSUPPORTED"
        result["phase"] = "parity"
        result["diagnostics"] = [_diagnostic(
            "UPSTREAM_PARITY_NOT_ESTABLISHED",
            "recipe does not select an enabled lowering parity mode",
            reference_mode=reference_mode,
        )]
        return result
    started = time.monotonic()
    try:
        phase = time.monotonic()
        if request["dsl"] == "tilelang":
            import tilelang  # noqa: F401
        else:
            import triton  # noqa: F401
        capsule = _capsule_module(request, f"pipeguard_capsule_{request['fingerprint'][:12]}")
        if reference_mode == "direct_original":
            reference = _direct_reference_module(request, f"pipeguard_reference_{request['fingerprint'][:12]}")
            reference_resolution = "original"
        else:
            reference = _reference_replay_module(
                request, f"pipeguard_reference_{request['fingerprint'][:12]}", output
            )
            reference_resolution = "capsule"
        result["timings"]["import"] = time.monotonic() - phase
    except Exception as error:
        result["status"] = "IMPORT_ERROR"
        result["phase"] = "import"
        result["diagnostics"] = [_diagnostic(
            "UPSTREAM_IMPORT_FAILED", str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=12)
        )]
        return result
    try:
        phase = time.monotonic()
        if request["dsl"] == "tilelang":
            tvm, reference_modules = _lower_tilelang(
                reference, dict(recipe), resolution=reference_resolution
            )
            tvm, capsule_modules = _lower_tilelang(capsule, dict(recipe), resolution="capsule")
            if [name for name, _ in reference_modules] != [name for name, _ in capsule_modules]:
                raise RuntimeError("device kernel symbol/count mismatch")
            raw_units = []
            for (name, reference_module), (_, capsule_module) in zip(reference_modules, capsule_modules):
                if not tvm.ir.structural_equal(reference_module, capsule_module, map_free_vars=False):
                    raise RuntimeError(f"TileLang structural parity mismatch for {name}")
                reference_raw = tvm.ir.save_json(reference_module)
                capsule_raw = tvm.ir.save_json(capsule_module)
                normalized_hash = sha256_text(str(tvm.ir.structural_hash(capsule_module, map_free_vars=False)))
                raw_units.append((name, capsule_raw, reference_raw, normalized_hash))
        else:
            reference_text = _lower_triton(reference, recipe, resolution=reference_resolution)
            capsule_text = _lower_triton(capsule, recipe, resolution="capsule")
            normalized_reference = _normalize_triton_ir(reference_text)
            normalized_capsule = _normalize_triton_ir(capsule_text)
            if normalized_reference != normalized_capsule:
                raise RuntimeError("Triton TTGIR parity mismatch after provenance-only normalization")
            raw_units = [("kernel", capsule_text, reference_text, sha256_text(normalized_capsule))]
        result["timings"]["lowering_and_parity"] = time.monotonic() - phase
    except Exception as error:
        result["status"] = "LOWERING_ERROR"
        result["phase"] = "lowering_parity"
        code = "UPSTREAM_PARITY_MISMATCH" if "parity" in str(error).lower() or "symbol/count" in str(error) else "UPSTREAM_LOWERING_FAILED"
        result["diagnostics"] = [_diagnostic(
            code, str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=20)
        )]
        return result
    if not raw_units:
        result["status"] = "PROFILE_SKIPPED"
        result["phase"] = "lowering_parity"
        result["diagnostics"] = [_diagnostic(
            "UPSTREAM_NO_DEVICE_KERNELS",
            "native lowering produced no device kernels for the selected target",
        )]
        result["timings"]["total"] = time.monotonic() - started
        return result
    snapshots: list[dict[str, Any]] = []
    phase = time.monotonic()
    for index, (symbol, raw, reference_raw, normalized_hash) in enumerate(raw_units):
        suffix = "device.tvm.json" if request["dsl"] == "tilelang" else "kernel.ttgir.mlir"
        raw_path = output / (suffix if len(raw_units) == 1 else f"{index:03d}-{symbol}.{suffix}")
        reference_path = output / (f"reference.{suffix}" if len(raw_units) == 1 else f"reference-{index:03d}-{symbol}.{suffix}")
        atomic_write(raw_path, raw)
        atomic_write(reference_path, reference_raw)
        result["artifacts"][f"raw_ir:{symbol}"] = {
            "path": raw_path.name,
            "sha256": sha256_text(raw),
            "reference_path": reference_path.name,
            "reference_sha256": sha256_text(reference_raw),
            "normalized_sha256": normalized_hash,
        }
        if request["dsl"] == "tilelang":
            snapshot, diagnostic = _capture_tilelang(request, raw_path, output)
        else:
            snapshot, diagnostic = _capture_triton(request, raw_path, output)
        if diagnostic is not None:
            result["status"] = (
                "PROFILE_SKIPPED" if diagnostic.get("code") in PROFILE_DIAGNOSTICS
                else "UNSUPPORTED"
            )
            result["phase"] = "capture"
            result["diagnostics"].append(diagnostic)
            continue
        assert snapshot is not None
        snapshot_path = output / ("snapshot.json" if len(raw_units) == 1 else f"snapshot-{index:03d}-{symbol}.json")
        snapshot_hash = _write_json(snapshot_path, snapshot)
        result["artifacts"][f"snapshot:{symbol}"] = {
            "path": snapshot_path.name,
            "sha256": snapshot_hash,
        }
        snapshots.append(snapshot)
    result["timings"]["capture"] = time.monotonic() - phase
    result["timings"]["total"] = time.monotonic() - started
    if len(snapshots) == len(raw_units):
        result["status"] = "CAPTURED"
        result["phase"] = "capture"
    elif snapshots:
        result["status"] = "UNSUPPORTED"
        result["diagnostics"].append(_diagnostic(
            "UPSTREAM_PARTIAL_CAPTURE_UNSUPPORTED",
            "not every lowered device kernel produced a strict snapshot",
            captured=len(snapshots), total=len(raw_units),
        ))
    return result


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-lower one locked upstream kernel recipe and run strict frontend capture")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        request = json.loads(arguments.request.read_text())
        result = run(request, arguments.output)
    except Exception as error:
        result = {
            "schema_version": RUN_RESULT_VERSION,
            "fingerprint": "0" * 64,
            "dsl": "tilelang",
            "candidate_id": "unknown",
            "recipe_id": "unknown",
            "status": "INFRA_ERROR",
            "phase": "worker",
            "diagnostics": [_diagnostic(
                "UPSTREAM_WORKER_FAILURE", str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=20)
            )],
            "artifacts": {},
            "timings": {},
        }
    atomic_write(arguments.output / "result.json", pretty_json(result))
    print(canonical_json(result))
    return 0 if result["status"] in TERMINAL_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
