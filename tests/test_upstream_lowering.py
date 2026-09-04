from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "corpus" / "upstream"
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", "/mnt/project_dft/xuwen/PipeGuard-asplos/tilelang"))
TRITON_ROOT = Path(os.environ.get("TRITON_ROOT", "/mnt/project_dft/xuwen/PipeGuard-asplos/triton"))
TILELANG_PYTHON = Path(os.environ.get("TILELANG_PYTHON", "/usr/local/miniconda3/envs/tl-pg/bin/python"))
TRITON_PYTHON = Path(os.environ.get("TRITON_PYTHON", "/usr/local/miniconda3/envs/triton-pg/bin/python"))
TRITON_PTXAS = Path(os.environ.get("TRITON_PTXAS_PATH", "/usr/local/cuda-12.9/bin/ptxas"))
sys.path.insert(0, str(ROOT / "tools"))
import upstream_worker  # noqa: E402


def _request(dsl: str, candidate_id: str, recipe_name: str, source_root: Path, fingerprint: str):
    corpus_name = "tilelang-2c31e071" if dsl == "tilelang" else "triton-61c02bcd"
    manifest = json.loads((CORPUS / corpus_name / "manifests" / f"{candidate_id}.json").read_text())
    recipe = json.loads((CORPUS / corpus_name / "recipes" / candidate_id / recipe_name).read_text())
    return {
        "fingerprint": fingerprint,
        "dsl": dsl,
        "corpus_root": str(CORPUS),
        "source_root": str(source_root),
        "manifest": manifest,
        "recipe": recipe,
    }


def _run_worker(tmp_path: Path, python: Path, request: dict, environment: dict[str, str] | None = None):
    request_path = tmp_path / f"{request['dsl']}-request.json"
    output = tmp_path / f"{request['dsl']}-output"
    request_path.write_text(json.dumps(request))
    env = dict(os.environ)
    env.update(environment or {})
    process = subprocess.run(
        [str(python), str(ROOT / "tools" / "upstream_worker.py"), "--request", str(request_path), "--output", str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stderr
    return json.loads((output / "result.json").read_text()), output


def test_curated_recipe_overlay_and_generated_recipes_validate():
    schema_root = CORPUS / "schemas"
    overlay = json.loads((CORPUS / "curated-recipes.json").read_text())
    overlay_schema = json.loads((schema_root / "curated-recipes-v1.schema.json").read_text())
    assert overlay["schema_version"] == overlay_schema["properties"]["schema_version"]["const"]
    assert set(overlay) == {"schema_version", "recipes"}
    recipe_schema = json.loads((schema_root / "recipe-v1.schema.json").read_text())
    for recipe in overlay["recipes"]:
        jsonschema.validate(recipe, recipe_schema)
    indexes = [
        json.loads((CORPUS / "tilelang-2c31e071" / "source-index.json").read_text()),
        json.loads((CORPUS / "triton-61c02bcd" / "source-index.json").read_text()),
    ]
    lowerable_total = sum(item["lowerable_recipes"] for index in indexes for item in index["candidates"])
    assert lowerable_total >= len(overlay["recipes"]) == 5
    indexed = {path.name for index in indexes for item in index["candidates"] for path in map(Path, item["recipe_paths"])}
    assert all(f"{recipe['recipe_id']}.json" in indexed for recipe in overlay["recipes"])


def test_worker_source_has_no_checker_dependency():
    source = (ROOT / "tools" / "upstream_worker.py").read_text()
    assert "pipeguard.checker" not in source
    assert "check_module" not in source


@pytest.mark.skipif(not TILELANG_PYTHON.exists() or not TILELANG_ROOT.exists(), reason="pinned TileLang environment unavailable")
def test_native_tilelang_upstream_lowering_reaches_profile_gate(tmp_path: Path):
    result, output = _run_worker(
        tmp_path,
        TILELANG_PYTHON,
        _request("tilelang", "dafb5bc213c7c107", "dafb5bc213c7c107-sm90a-default.json", TILELANG_ROOT, "2" * 64),
    )
    assert result["status"] == "PROFILE_SKIPPED"
    assert result["diagnostics"][0]["code"] == "TILELANG_MULTI_CTA_UNSUPPORTED"
    artifact = result["artifacts"]["raw_ir:main_kernel"]
    assert (output / artifact["path"]).exists()
    assert (output / artifact["reference_path"]).exists()
    assert len(artifact["normalized_sha256"]) == 64


@pytest.mark.skipif(not TILELANG_PYTHON.exists() or not TILELANG_ROOT.exists(), reason="pinned TileLang environment unavailable")
def test_native_tilelang_bootstraps_tvm_and_rejects_empty_device_module(tmp_path: Path):
    candidate_id = "0ed9a4b7c6fe013d"
    result, _output = _run_worker(
        tmp_path,
        TILELANG_PYTHON,
        _request(
            "tilelang",
            candidate_id,
            f"{candidate_id}-sm90a-static-primfunc.json",
            TILELANG_ROOT,
            "4" * 64,
        ),
    )
    assert result["status"] == "PROFILE_SKIPPED"
    assert result["diagnostics"][0]["code"] == "UPSTREAM_NO_DEVICE_KERNELS"
    assert result["artifacts"] == {}


@pytest.mark.skipif(not TRITON_PYTHON.exists() or not TRITON_ROOT.exists() or not TRITON_PTXAS.exists(), reason="pinned Triton environment unavailable")
def test_native_triton_upstream_lowering_reaches_alias_gate(tmp_path: Path):
    result, output = _run_worker(
        tmp_path,
        TRITON_PYTHON,
        _request("triton", "e2bf6ebf7bbe9806", "e2bf6ebf7bbe9806-sm90-block128.json", TRITON_ROOT, "3" * 64),
        {
            "TRITON_PTXAS_PATH": str(TRITON_PTXAS),
            "TRITON_CACHE_DIR": str(tmp_path / "triton-cache"),
        },
    )
    assert result["status"] == "UNSUPPORTED"
    assert result["diagnostics"][0]["code"] == "TRITON_ALIAS_UNRESOLVED"
    artifact = result["artifacts"]["raw_ir:kernel"]
    assert (output / artifact["path"]).exists()
    assert artifact["sha256"] == artifact["reference_sha256"]


def _unit_request(dsl: str) -> dict:
    return {
        "fingerprint": "1" * 64,
        "dsl": dsl,
        "corpus_root": str(CORPUS),
        "source_root": str(ROOT),
        "manifest": {
            "candidate_id": "unit-candidate",
            "upstream": {"qualname": "kernel"},
        },
        "recipe": {
            "recipe_id": "unit-recipe",
            "lowerable": True,
            "reference": {"mode": "direct_original"},
        },
    }


def _isolate_unit_lowering(monkeypatch: pytest.MonkeyPatch, dsl: str) -> None:
    monkeypatch.setitem(sys.modules, dsl, types.ModuleType(dsl))
    monkeypatch.setattr(upstream_worker, "_capsule_module", lambda *_args: object())
    monkeypatch.setattr(upstream_worker, "_direct_reference_module", lambda *_args: object())


def _forbid_capture(*_args, **_kwargs):
    pytest.fail("parity or lowering failure entered capture")


def test_compiler_error_message_containing_parity_is_lowering_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _isolate_unit_lowering(monkeypatch, "triton")

    def fail_lowering(*_args, **_kwargs):
        raise RuntimeError("compiler parity optimization failed")

    monkeypatch.setattr(upstream_worker, "_lower_triton", fail_lowering)
    monkeypatch.setattr(upstream_worker, "_capture_triton", _forbid_capture)
    result = upstream_worker.run(_unit_request("triton"), tmp_path)

    assert result["status"] == "LOWERING_ERROR"
    assert result["diagnostics"][0]["code"] == "UPSTREAM_LOWERING_FAILED"
    assert result["diagnostics"][0]["details"]["error_type"] == "RuntimeError"
    assert result["artifacts"] == {}


def test_triton_normalized_mismatch_preserves_both_raw_ir_without_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _isolate_unit_lowering(monkeypatch, "triton")

    def lower_triton(_module, _recipe, *, resolution: str):
        return (
            "module { reference }\n"
            if resolution == "original"
            else "module { capsule }\n"
        )

    monkeypatch.setattr(upstream_worker, "_lower_triton", lower_triton)
    monkeypatch.setattr(upstream_worker, "_capture_triton", _forbid_capture)
    result = upstream_worker.run(_unit_request("triton"), tmp_path)

    assert result["status"] == "LOWERING_ERROR"
    assert result["diagnostics"][0]["code"] == "UPSTREAM_PARITY_MISMATCH"
    assert result["diagnostics"][0]["details"]["error_type"] == "UpstreamParityError"
    artifact = result["artifacts"]["raw_ir:kernel"]
    for path_key, hash_key in (("path", "sha256"), ("reference_path", "reference_sha256")):
        artifact_path = tmp_path / artifact[path_key]
        assert artifact_path.is_file()
        assert upstream_worker.sha256_text(artifact_path.read_text()) == artifact[hash_key]
    assert not list(tmp_path.glob("snapshot*.json"))


class _FakeTileLangIR:
    @staticmethod
    def save_json(module: str) -> str:
        return module

    @staticmethod
    def structural_hash(module: str, *, map_free_vars: bool) -> int:
        assert map_free_vars is False
        return len(module)

    @staticmethod
    def structural_equal(left: str, right: str, *, map_free_vars: bool) -> bool:
        assert map_free_vars is False
        return left == right


class _FakeTVM:
    ir = _FakeTileLangIR()


def test_tilelang_structural_mismatch_preserves_both_raw_ir_without_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _isolate_unit_lowering(monkeypatch, "tilelang")

    def lower_tilelang(_module, _recipe, *, resolution: str):
        module = "reference raw ir" if resolution == "original" else "capsule raw ir"
        return _FakeTVM(), [("kernel", module)]

    monkeypatch.setattr(upstream_worker, "_lower_tilelang", lower_tilelang)
    monkeypatch.setattr(upstream_worker, "_capture_tilelang", _forbid_capture)
    result = upstream_worker.run(_unit_request("tilelang"), tmp_path)

    assert result["diagnostics"][0]["code"] == "UPSTREAM_PARITY_MISMATCH"
    artifact = result["artifacts"]["raw_ir:kernel"]
    assert (tmp_path / artifact["path"]).read_text() == "capsule raw ir"
    assert (tmp_path / artifact["reference_path"]).read_text() == "reference raw ir"
    assert not list(tmp_path.glob("snapshot*.json"))


def test_tilelang_symbol_count_mismatch_preserves_every_raw_ir_without_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _isolate_unit_lowering(monkeypatch, "tilelang")

    def lower_tilelang(_module, _recipe, *, resolution: str):
        modules = (
            [("first", "reference first"), ("second", "reference second")]
            if resolution == "original"
            else [("only", "capsule only")]
        )
        return _FakeTVM(), modules

    monkeypatch.setattr(upstream_worker, "_lower_tilelang", lower_tilelang)
    monkeypatch.setattr(upstream_worker, "_capture_tilelang", _forbid_capture)
    result = upstream_worker.run(_unit_request("tilelang"), tmp_path)

    assert result["diagnostics"][0]["code"] == "UPSTREAM_PARITY_MISMATCH"
    assert set(result["artifacts"]) == {
        "raw_ir:reference:0:first",
        "raw_ir:reference:1:second",
        "raw_ir:capsule:0:only",
    }
    for artifact in result["artifacts"].values():
        artifact_path = tmp_path / artifact["path"]
        assert artifact_path.is_file()
        assert upstream_worker.sha256_text(artifact_path.read_text()) == artifact["sha256"]
    assert not list(tmp_path.glob("snapshot*.json"))
