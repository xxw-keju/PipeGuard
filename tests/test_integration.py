"""Installation-facing API contracts; independent of compiler availability."""
import hashlib
import json
import signal
import subprocess

import pytest

from pipeguard import integration


@pytest.mark.parametrize("kwargs", [
    {"launch_grid": (0,)}, {"launch_grid": (1,), "selected_cta": (1, 0, 0)},
    {"launch_grid": (1,), "timeout": 0}, {"launch_grid": (1,), "timeout": float("inf")},
])
def test_invalid_invocation_rejected_before_worker(kwargs):
    with pytest.raises(ValueError):
        integration.check_tilelang_ir("{}", **kwargs)


def test_worker_evidence_is_bound_to_exact_ir(tmp_path, monkeypatch):
    from pipeguard.capture import tilelang_capture

    seen = {}
    def capture(path, source, command, descriptors, bindings, **kwargs):
        seen.update(ir=path.read_text(), descriptors=json.loads(descriptors.read_text()),
                    bindings=json.loads(bindings.read_text()), kwargs=kwargs)
        raise tilelang_capture.CaptureError("EXACT_INPUT_REQUIRED", "unresolved test operand")
    monkeypatch.setattr(tilelang_capture, "capture_file", capture)
    request = dict(ir_json='{"test":"exact IR"}', source="test", launch_grid=[2, 1, 1],
                   selected_cta=[1, 0, 0], descriptors={"d": {"shape": [64]}},
                   scalar_values={"n": 64}, pointer_table_allocations={}, buffer_values={})
    report = integration._worker(request, tmp_path)
    assert report["verdict"] == "UNSUPPORTED"
    assert report["diagnostics"][0]["code"] == "EXACT_INPUT_REQUIRED"
    digest = hashlib.sha256(request["ir_json"].encode()).hexdigest()
    assert seen["bindings"]["source_ir_sha256"] == digest
    assert seen["descriptors"]["source_ir_sha256"] == digest
    assert seen["bindings"]["scalar_values"] == {"n": 64}
    assert seen["kwargs"]["selected_cta"] == [1, 0, 0]


def test_timeout_kills_process_group_and_never_returns_safe(monkeypatch):
    calls = []
    class Process:
        pid = 12345
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("test", timeout)
            calls.append("reaped")
    monkeypatch.setattr(integration.subprocess, "Popen", Process)
    monkeypatch.setattr(integration.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    report = integration.check_tilelang_ir("{}", launch_grid=(1,), timeout=0.1)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["diagnostics"][0]["code"] == "CAPTURE_CHECK_TIMEOUT"
    assert calls == [(12345, signal.SIGKILL), "reaped"]
    assert report["scope"]["launch_grid"] == [1, 1, 1]


def test_worker_failure_is_not_unsupported_or_safe(monkeypatch):
    class Process:
        returncode = 1
        def __init__(self, command, **kwargs):
            kwargs["stderr"].write("worker failure")
        def wait(self, timeout=None):
            return self.returncode
    monkeypatch.setattr(integration.subprocess, "Popen", Process)
    report = integration.check_tilelang_ir("{}", launch_grid=(1,))
    assert report["verdict"] == "INTERNAL_ERROR"
    assert "worker failure" in report["diagnostics"][0]["message"]
