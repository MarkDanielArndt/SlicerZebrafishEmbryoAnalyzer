"""
Asynchronous QProcess-based inference runner for ZebrafishEmbryoAnalyzer.

Launches inference_worker.py as a standalone subprocess, streams progress via
stdout, and delivers results back to the main thread via on_finished callback.
No Python background threads are created.

Architecture boundary: this module imports qt (PythonQt) and must only be
imported from the Slicer main thread.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


TERMINAL_STATES = {"succeeded", "cancelled", "failed", "disposed"}


class InferenceController:
    """Signal-driven subprocess inference controller.

    Mirrors ModelDownloadController pattern: constructor stores configuration,
    start() launches the process, callbacks fire on the Qt event loop.
    """

    def __init__(
        self,
        image_paths,
        model_id,
        params,
        originals,
        on_finished,
        on_progress=None,
        qt_module=None,
        slicer_module=None,
        process_factory=None,
        parent=None,
    ):
        self.image_paths = list(image_paths)
        self.model_id = model_id
        self.params = dict(params)
        self.originals = list(originals) if originals is not None else []
        self.on_finished = on_finished
        self.on_progress = on_progress
        self.qt = qt_module
        self.slicer = slicer_module
        self.process_factory = process_factory
        self.parent = parent

        self.state = "idle"
        self._process = None
        self._tmp_dir = None
        self._request_json_path = None
        self._result_json_path = None
        self._arrays_dir = None
        self._finished_called = False
        self._stdout_buffer = ""
        self.cancelled = False
        self.disposed = False
        self.results = []
        self.exit_code = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Write request, launch worker subprocess, return self."""
        self._ensure_modules()
        self._tmp_dir = tempfile.mkdtemp(prefix="zebrafish_run_")
        self._request_json_path = os.path.join(self._tmp_dir, "request.json")
        self._result_json_path = os.path.join(self._tmp_dir, "result.json")
        self._arrays_dir = os.path.join(self._tmp_dir, "arrays")

        request = {
            "protocol_version": 1,
            "model_id": self.model_id,
            "image_paths": self.image_paths,
            "params": self.params,
            "result_json": self._result_json_path,
            "arrays_dir": self._arrays_dir,
        }
        try:
            with open(self._request_json_path, "w", encoding="utf-8") as fh:
                json.dump(request, fh)
        except Exception as exc:
            self._finish_once("failed", False, f"Cannot write request: {exc}")
            return self

        python_exe = sys.executable or ""
        if not python_exe:
            _is_win = sys.platform == "win32"
            _scripts = "Scripts" if _is_win else "bin"
            _names = ("python.exe",) if _is_win else ("python3", "python")
            for _name in _names:
                candidate = Path(sys.prefix) / _scripts / _name
                if candidate.exists():
                    python_exe = str(candidate)
                    break
        if not python_exe:
            self._finish_once("failed", False, "Cannot find Python executable")
            return self

        worker_script = str(Path(__file__).parent / "inference_worker.py")

        if self.process_factory is not None:
            self._process = self.process_factory()
        else:
            self._process = self.qt.QProcess(self.parent)

        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)

        # Ensure ZebrafishEmbryoAnalyzer/ is on sys.path in the subprocess
        _zebrafish_lib_root = str(Path(__file__).parent.parent)
        try:
            env = self.qt.QProcessEnvironment.systemEnvironment()
            existing = env.value("PYTHONPATH", "")
            new_pp = _zebrafish_lib_root + os.pathsep + existing if existing else _zebrafish_lib_root
            env.insert("PYTHONPATH", new_pp)
            self._process.setProcessEnvironment(env)
        except Exception:
            pass  # QProcessEnvironment unavailable; sys.executable path may still work

        self._process.start(python_exe, ["-u", worker_script, self._request_json_path])
        self.state = "running"
        return self

    def cancel(self):
        """Terminate the subprocess and transition to cancelled."""
        if self.state in TERMINAL_STATES:
            return
        self.cancelled = True
        process_ref = self._process  # capture before _finish_once nulls it
        if process_ref is not None:
            try:
                process_ref.terminate()
            except Exception:
                pass
            try:
                def _kill_if_alive():
                    try:
                        process_ref.kill()
                    except Exception:
                        pass
                self.qt.QTimer.singleShot(3000, _kill_if_alive)
            except Exception:
                pass
        self._finish_once("cancelled", False, None)

    def dispose(self):
        """Suppress the on_finished callback and cancel silently."""
        if self.state in TERMINAL_STATES:
            self.disposed = True
            return
        self.disposed = True
        self.cancel()

    # ------------------------------------------------------------------
    # Process signal handlers
    # ------------------------------------------------------------------

    def _on_ready_read(self):
        if self._process is None:
            return
        try:
            raw = self._process.readAllStandardOutput()
            if hasattr(raw, "data"):
                data = bytes(raw.data())
            else:
                data = bytes(raw)
            self._stdout_buffer += data.decode("utf-8", errors="replace")
        except Exception:
            return

        lines = self._stdout_buffer.split("\n")
        self._stdout_buffer = lines[-1]
        for line in lines[:-1]:
            line = line.strip()
            if line.startswith("PROGRESS "):
                try:
                    _, fraction = line.split(" ", 1)
                    i_str, n_str = fraction.split("/")
                    i, n = int(i_str), int(n_str)
                    if self.on_progress is not None:
                        self.on_progress(i, n)
                except Exception:
                    pass

    def _on_finished(self, exit_code, exit_status=None):
        if self.cancelled:
            self._finish_once("cancelled", False, None)
            return

        self.exit_code = exit_code

        if exit_code == 0:
            try:
                with open(self._result_json_path, "r", encoding="utf-8") as fh:
                    result_data = json.load(fh)
                worker_results = result_data.get("results", [])
                internal_results = self._convert_results(worker_results)
                self._merge_originals(internal_results, self.originals)
                self.results = internal_results
                self._finish_once("succeeded", True, None)
            except Exception as exc:
                self._finish_once("failed", False, f"Cannot read results: {exc}")
        else:
            message = self._read_error_message()
            self._finish_once("failed", False, message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_modules(self):
        if self.qt is None:
            import qt
            self.qt = qt
        if self.slicer is None:
            try:
                import slicer
                self.slicer = slicer
            except ImportError:
                self.slicer = None

    def _read_error_message(self):
        """Try result JSON first, then stderr, then return a generic message."""
        try:
            if self._result_json_path and os.path.exists(self._result_json_path):
                with open(self._result_json_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                msg = data.get("error_message", "")
                if msg:
                    return msg
        except Exception:
            pass
        try:
            if self._process is not None:
                raw = self._process.readAllStandardError()
                if hasattr(raw, "data"):
                    data = bytes(raw.data())
                else:
                    data = bytes(raw)
                text = data.decode("utf-8", errors="replace").strip()
                if text:
                    return text[:1000]
        except Exception:
            pass
        return "Worker process failed"

    def _convert_results(self, worker_results):
        """Convert wire-format result dicts back to internal widget format."""
        import numpy as np
        internal = []
        for r in worker_results:
            ir = {
                "filename": r.get("filename"),
                "image_path": r.get("image_path"),
                "length": r.get("length_um"),
                "curvature": r.get("curvature_class"),
                "ratio": r.get("length_straight_ratio"),
                "eye_area": r.get("eye_area_um2"),
                "eye_diameter": r.get("eye_diameter_um"),
                "spacing": r.get("spacing"),
                "error": r.get("error"),
                "original": None,
                "mask": None,
                "grown": None,
                "eye_mask": None,
                "path_points": None,
                "straight_line_points": None,
            }
            npz_path = r.get("arrays_npz")
            if npz_path and os.path.exists(npz_path):
                try:
                    arrays = np.load(npz_path, allow_pickle=False)
                    for k in ("mask", "grown", "eye_mask",
                              "path_points", "straight_line_points"):
                        if k in arrays:
                            ir[k] = arrays[k]
                except Exception:
                    pass
            internal.append(ir)
        return internal

    def _merge_originals(self, results, originals):
        """Re-attach original images captured before worker was launched."""
        for i, r in enumerate(results):
            if i < len(originals) and originals[i] is not None:
                r["original"] = originals[i]

    def _finish_once(self, state, success, message):
        """Idempotent terminal transition: disconnect signals, cleanup, fire callback."""
        if self._finished_called or self.state in TERMINAL_STATES:
            return
        self._finished_called = True
        self.state = state

        if self._process is not None:
            try:
                self._process.readyReadStandardOutput.disconnect(self._on_ready_read)
            except Exception:
                pass
            try:
                self._process.finished.disconnect(self._on_finished)
            except Exception:
                pass
            self._process = None

        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            try:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
            except Exception:
                pass
        self._tmp_dir = None

        if not success:
            self.results = []

        callback = self.on_finished
        self.on_finished = None
        if callback is not None and not self.disposed:
            callback(success, state, message, self)


def start_inference(image_paths, model_id, params, originals, on_finished,
                    on_progress=None, **kwargs):
    """Create, configure, and start an InferenceController. Returns the controller."""
    controller = InferenceController(
        image_paths, model_id, params, originals, on_finished, on_progress, **kwargs
    )
    return controller.start()
