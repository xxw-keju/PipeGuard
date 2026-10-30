"""Local-buffer loop counters must advance capture without hiding real cycles."""
import os
from pathlib import Path
import subprocess

import pytest

from test_frontend_capture_tools import ROOT, TILELANG_PYTHON


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang unavailable")
@pytest.mark.parametrize("walker", ["grouped", "lane"])
@pytest.mark.parametrize("update", ["advance", "stationary", "cycle"])
def test_local_counter_while_progress_and_cycles(tmp_path: Path, walker: str, update: str):
    script = tmp_path / "while_state.py"
    script.write_text('''import sys
from tilelang import tvm
from tilelang_capture import TileLangCapture, LaneWalker, CaptureError

T = tvm.tirx
mode, update = sys.argv[1:]
counter = T.decl_buffer((1,), "int32", name="counter", scope="local")
output = T.decl_buffer((10,), "int32", name="output")
value = counter[0] + 1 if update == "advance" else (1 - counter[0] if update == "cycle" else counter[0])
body = T.SeqStmt([
    T.AllocBuffer(counter),
    T.BufferStore(counter, 0, [0]),
    T.While(counter[0] < 10, T.SeqStmt([
        T.BufferStore(output, 1, [counter[0]]),
        T.BufferStore(counter, value, [0]),
    ])),
])
func = T.PrimFunc([output.data], body, buffer_map={output.data: output})
func = func.with_attr("target", tvm.target.Target({"kind": "cuda", "arch": "sm_90a"}))
func = func.with_attr("thread_extent", {"threadIdx.x": 1}).with_attr("tirx.noalias", True)
capture = TileLangCapture(tvm, T, tvm.IRModule({"main": func}), "while-test", "test", "0" * 64)
try:
    if mode == "grouped":
        capture.capture()
    else:
        lane = LaneWalker(capture, 0)
        lane.walk(func.body, ("root",))
        lane.finalize()
except CaptureError as error:
    assert update != "advance", error.diagnostic()
    assert error.code == "TILELANG_NONFINITE_CONTROL_FLOW", error.diagnostic()
    assert error.details["iteration"] == (2 if update == "cycle" else 1)
else:
    assert update == "advance", "non-terminating loop was accepted"
    offsets = {r["offset"] for r in capture.regions.values() if r["address_space"] == "GLOBAL"}
    assert offsets == set(range(0, 40, 4)), offsets
    assert capture.static_expansions == 10
''')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "tools"))
    process = subprocess.run([str(TILELANG_PYTHON), str(script), walker, update],
                             env=env, text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
