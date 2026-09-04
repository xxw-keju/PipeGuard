from __future__ import annotations

import json
import os
import subprocess
import sys
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
