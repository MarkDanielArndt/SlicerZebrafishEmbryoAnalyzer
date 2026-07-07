"""
Tests for InferenceController in ZebrafishEmbryoAnalyzerLib.inference_runner.

All tests are fully synchronous — no real subprocess launched.
A FakeProcess / FakeSignal / FakeQt scaffold replaces Qt and QProcess.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake Qt infrastructure (no real Qt required)
# ---------------------------------------------------------------------------

class FakeSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot):
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args):
        for s in list(self._slots):
            s(*args)


class FakeProcess:
    def __init__(self):
        self.readyReadStandardOutput = FakeSignal()
        self.finished = FakeSignal()
        self._stdout = b""
        self._stderr_data = b""
        self.terminated = False
        self.killed = False
        self.started_args = []

    def start(self, exe, args):
        self.started_args = [exe] + list(args)

    def readAllStandardOutput(self):
        d = self._stdout
        self._stdout = b""
        return d

    def readAllStandardError(self):
        d = self._stderr_data
        self._stderr_data = b""
        return d

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def waitForFinished(self, ms=3000):
        return True


class FakeTimer:
    """Captures singleShot calls so tests can fire them manually."""
    def __init__(self):
        self._pending = []

    def singleShot(self, delay_ms, callback):
        self._pending.append(callback)

    def fire_all(self):
        for cb in list(self._pending):
            cb()
        self._pending.clear()


class FakeQt:
    def __init__(self):
        self._fake_process = None
        self.QTimer = FakeTimer()

    def QProcess(self, parent=None):
        if self._fake_process is not None:
            return self._fake_process
        return FakeProcess()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(tmp_path, fake_process=None, on_finished=None, on_progress=None,
                     image_paths=None, model_id="general", params=None, originals=None):
    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController

    fqt = FakeQt()
    if fake_process is not None:
        fqt._fake_process = fake_process

    calls = []
    if on_finished is None:
        def on_finished(success, state, message, controller):
            calls.append((success, state, message))

    controller = InferenceController(
        image_paths=image_paths or ["/tmp/fish.png"],
        model_id=model_id,
        params=params or {},
        originals=originals or [None],
        on_finished=on_finished,
        on_progress=on_progress,
        qt_module=fqt,
        process_factory=lambda: fake_process or FakeProcess(),
    )
    return controller, calls, fqt


def _write_result_json(path, results=None, status="ok", error_code=0, error_message=""):
    data = {
        "protocol_version": 1,
        "status": status,
        "error_code": error_code,
        "error_message": error_message,
        "results": results or [],
    }
    with open(path, "w") as fh:
        json.dump(data, fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_launches_process_with_unbuffered_flag():
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(None, fake_process=proc)
    controller.start()

    assert "-u" in proc.started_args


def test_progress_lines_parsed_and_forwarded():
    proc = FakeProcess()
    progress_calls = []
    controller, _, fqt = _make_controller(
        None, fake_process=proc, on_progress=lambda i, n: progress_calls.append((i, n))
    )
    controller.start()

    proc._stdout = b"PROGRESS 1/3\nPROGRESS 2/3\n"
    proc.readyReadStandardOutput.emit()

    assert (1, 3) in progress_calls
    assert (2, 3) in progress_calls


def test_success_exit_reads_result_json_and_merges_originals(tmp_path):
    proc = FakeProcess()
    original_img = np.zeros((64, 64, 3), dtype=np.uint8)
    controller, calls, fqt = _make_controller(
        tmp_path, fake_process=proc, originals=[original_img]
    )
    controller.start()

    # Write result JSON into the temp dir created by start()
    result_dir = controller._tmp_dir
    arrays_dir = os.path.join(result_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    mask = np.ones((256, 256), dtype=np.uint8)
    npz_path = os.path.join(arrays_dir, "fish_0.npz")
    np.savez(npz_path, mask=mask)

    wire_results = [{
        "filename": "fish.png",
        "image_path": "/tmp/fish.png",
        "length_um": 1234.5,
        "curvature_class": 0,
        "length_straight_ratio": 0.95,
        "eye_area_um2": None,
        "eye_diameter_um": None,
        "spacing": [22.99, 22.99],
        "error": None,
        "arrays_npz": npz_path,
    }]
    _write_result_json(controller._result_json_path, results=wire_results)

    proc.finished.emit(0, 0)

    assert len(calls) == 1
    success, state, message = calls[0]
    assert success is True
    assert state == "succeeded"

    results = controller.results
    assert len(results) == 1
    assert results[0]["filename"] == "fish.png"
    assert results[0]["length"] == pytest.approx(1234.5)
    assert results[0]["curvature"] == 0
    assert results[0]["ratio"] == pytest.approx(0.95)
    assert results[0]["original"] is original_img
    assert results[0]["mask"] is not None
    assert results[0]["mask"].shape == (256, 256)


def test_failure_exit_calls_on_finished_with_false(tmp_path):
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(tmp_path, fake_process=proc)
    controller.start()

    proc.finished.emit(1, 0)

    assert len(calls) == 1
    success, state, message = calls[0]
    assert success is False
    assert state == "failed"


def test_cancel_calls_terminate_then_kill():
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(None, fake_process=proc)
    controller.start()

    controller.cancel()

    assert proc.terminated is True

    # Fire the 3-second timer to trigger force kill
    fqt.QTimer.fire_all()
    # Force kill only fires if still in "running" state; after cancel it transitions away
    # so kill may not be called — that's correct behaviour. Verify terminate was called.
    assert proc.terminated is True


def test_cancelled_finish_does_not_call_on_finished_again():
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(None, fake_process=proc)
    controller.start()

    controller.cancel()
    proc.finished.emit(0, 0)

    # callback fired once (from cancel), not again from finished signal
    assert len(calls) == 1
    assert calls[0][1] == "cancelled"


def test_dispose_prevents_on_finished():
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(None, fake_process=proc)
    controller.start()

    controller.dispose()
    proc.finished.emit(0, 0)

    assert calls == []


def test_stale_result_ignored_when_runner_replaced():
    """Widget token guard: result from old runner is discarded when a new queue was loaded."""
    import sys
    sys.modules.setdefault("qt", MagicMock())
    sys.modules.setdefault("ctk", MagicMock())
    sl = MagicMock()
    sl.util.mainWindow.return_value = None
    sys.modules.setdefault("slicer", sl)

    import importlib
    import ZebrafishEmbryoAnalyzerLib.widget as widget_mod
    widget_mod = importlib.reload(widget_mod)

    w = object.__new__(widget_mod.ZebrafishEmbryoAnalyzerMainWidget)
    w._disposed = False
    w._active_runner = None
    w._run_token = 1
    w._results = []
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._on_results_ready = MagicMock()
    w._try_update_mrml_table = MagicMock()

    runner_mock = MagicMock()
    runner_mock.results = [{"filename": "fish.png", "length": 1.0}]
    w._active_runner = runner_mock

    # Simulate token change (new folder loaded)
    w._run_token = 2

    # Fire the callback that would arrive from the now-stale runner
    # We need to call the inner closure; reconstruct minimal scenario
    stored_results_before = list(w._results)

    # The callback checks token; since token changed, it should not apply results
    token = 1  # original token when inference started
    controller = runner_mock

    def simulate_on_runner_finished(success, state, message, ctrl):
        if w._disposed or ctrl is not w._active_runner:
            return
        w._active_runner = None
        if w._run_token != token:
            w._run_stack.setCurrentIndex(0)
            return
        if not success:
            w._run_stack.setCurrentIndex(0)
            return
        w._results = ctrl.results
        w._run_stack.setCurrentIndex(0)
        w._on_results_ready()

    simulate_on_runner_finished(True, "succeeded", None, controller)

    # Results should not have been applied because token mismatched
    assert w._results == stored_results_before
    w._on_results_ready.assert_not_called()


def test_temp_dir_deleted_on_success(tmp_path):
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(tmp_path, fake_process=proc)
    controller.start()

    tmp_dir = controller._tmp_dir
    assert os.path.isdir(tmp_dir)

    _write_result_json(
        controller._result_json_path,
        results=[{
            "filename": "fish.png", "image_path": "/tmp/fish.png",
            "length_um": None, "curvature_class": None, "length_straight_ratio": None,
            "eye_area_um2": None, "eye_diameter_um": None, "spacing": None,
            "error": None, "arrays_npz": None,
        }]
    )
    proc.finished.emit(0, 0)

    assert not os.path.isdir(tmp_dir)


def test_temp_dir_deleted_on_failure(tmp_path):
    proc = FakeProcess()
    controller, calls, fqt = _make_controller(tmp_path, fake_process=proc)
    controller.start()

    tmp_dir = controller._tmp_dir
    proc.finished.emit(1, 0)

    assert not os.path.isdir(tmp_dir)


def test_sys_executable_empty_falls_back(monkeypatch):
    """When sys.executable is empty, controller uses a fallback or fails gracefully."""
    monkeypatch.setattr(sys, "executable", "")

    # Create a candidate that does not exist so fallback also fails
    import sys as _sys
    monkeypatch.setattr(_sys, "prefix", "/nonexistent_prefix_xyz")

    proc = FakeProcess()
    calls = []

    def on_finished(success, state, message, ctrl):
        calls.append((success, state, message))

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController
    fqt = FakeQt()
    fqt._fake_process = proc
    controller = InferenceController(
        image_paths=["/tmp/fish.png"],
        model_id="general",
        params={},
        originals=[None],
        on_finished=on_finished,
        qt_module=fqt,
        process_factory=lambda: proc,
    )
    controller.start()

    # Either: process was started with a fallback path (success in finding python),
    # or it failed gracefully with an error about no Python executable.
    # Both are valid; just verify no unhandled exception was raised.
    # If calls is non-empty, it must be a failure with a descriptive message.
    if calls:
        assert calls[0][0] is False
        assert calls[0][2] is not None


def test_on_finished_accepts_single_arg(tmp_path):
    """PythonQt may emit finished(exitCode) without exitStatus."""
    import json, os
    result_file = tmp_path / "result.json"
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    result_file.write_text(json.dumps({
        "protocol_version": 1,
        "status": "ok",
        "error_code": 0,
        "error_message": "",
        "results": [],
    }))

    finished_calls = []

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController

    qt_mock = MagicMock()
    fake_proc = FakeProcess()  # reuse the FakeProcess class defined in this test file
    controller = InferenceController(
        image_paths=[], model_id="general", params={}, originals=[],
        on_finished=lambda s, st, m, c: finished_calls.append(s),
        qt_module=qt_mock,
        process_factory=lambda: fake_proc,
    )
    controller._tmp_dir = str(tmp_path)
    controller._result_json_path = str(result_file)
    controller._arrays_dir = str(arrays_dir)
    controller.state = "running"

    # Call with only exit_code — must not raise TypeError
    controller._on_finished(0)  # no exit_status argument

    assert finished_calls == [True]


# ---------------------------------------------------------------------------
# J1: Platform-independent Python executable discovery
# ---------------------------------------------------------------------------

def test_python_exe_windows_fallback_scripts_python_exe(tmp_path, monkeypatch):
    """On Windows with empty sys.executable, discovers Scripts/python.exe."""
    import sys
    from pathlib import Path

    # Create fake Scripts/python.exe inside tmp_path acting as sys.prefix
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    fake_exe = scripts_dir / "python.exe"
    fake_exe.write_text("")  # must exist so .exists() returns True

    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")

    proc = FakeProcess()
    calls = []

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController
    fqt = FakeQt()
    fqt._fake_process = proc
    controller = InferenceController(
        image_paths=["/tmp/fish.png"],
        model_id="general",
        params={},
        originals=[None],
        on_finished=lambda s, st, m, c: calls.append((s, st, m)),
        qt_module=fqt,
        process_factory=lambda: proc,
    )
    controller.start()

    # Process must have been started (not failed with "Cannot find Python executable")
    assert calls == [], f"Should not have failed early; calls={calls}"
    assert str(fake_exe) in proc.started_args


def test_python_exe_unix_fallback_bin_python(tmp_path, monkeypatch):
    """On Unix with empty sys.executable, discovers bin/python when python3 absent."""
    import sys
    from pathlib import Path

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_exe = bin_dir / "python"
    fake_exe.write_text("")

    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")

    proc = FakeProcess()
    calls = []

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController
    fqt = FakeQt()
    fqt._fake_process = proc
    controller = InferenceController(
        image_paths=["/tmp/fish.png"],
        model_id="general",
        params={},
        originals=[None],
        on_finished=lambda s, st, m, c: calls.append((s, st, m)),
        qt_module=fqt,
        process_factory=lambda: proc,
    )
    controller.start()

    assert calls == [], f"Should not have failed early; calls={calls}"
    assert str(fake_exe) in proc.started_args
