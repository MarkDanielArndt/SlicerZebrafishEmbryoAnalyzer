"""
Tests for ZebrafishAnalysisLogic — the Slicer ScriptedLoadableModuleLogic subclass.

All tests run outside Slicer via subprocess with a minimal slicer/qt stub so the
module-level imports in ZebrafishAnalysis.py don't fail.  This isolates the logic
class API contract from widget construction and from Slicer's runtime.
"""

import os
import sys
import textwrap
import subprocess

import pytest


_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishAnalysis"
)

_SLICER_STUB = """
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

# ScriptedLoadableModuleWidget must be a distinct class (not object itself) and
# VTKObservationMixin must also be a distinct class, so that
# class ZebrafishAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin)
# can resolve its MRO — Python C3 rejects (object, SomeSubclassOfObject).
class _BaseWidget:
    pass

class _VTKObservationMixinStub:
    def addObserver(self, *a, **kw): pass
    def removeObservers(self, *a, **kw): pass
    def hasObserver(self, *a, **kw): return False

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_BaseWidget,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
# VTKObservationMixin lives in slicer.util in the real Slicer runtime.
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=_VTKObservationMixinStub,
)
# vtk is imported at the top of ZebrafishAnalysis.py
_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""


def _run(code: str) -> subprocess.CompletedProcess:
    full = _SLICER_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# Logic class API
# ---------------------------------------------------------------------------

def test_logic_class_can_be_instantiated():
    """ZebrafishAnalysisLogic can be instantiated without Slicer runtime."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_class_exposes_analysis_methods():
    """ZebrafishAnalysisLogic must expose run_analysis, detect_scalebar, preload_models."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        for name in ("run_analysis", "detect_scalebar", "preload_models",
                     "apply_manual_correction", "revert_manual_correction"):
            assert callable(getattr(logic, name, None)), f"missing method: {name}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Delegation — observable behavior of logic methods
# ---------------------------------------------------------------------------

def test_logic_run_analysis_delegates_to_lib():
    """logic.run_analysis must delegate to ZebrafishAnalysisLib.logic.analyse_images."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[{"filename": "x.png"}]) as mock:
            result = logic.run_analysis(["/x.png"], {"length": True})
        assert mock.called, "analyse_images was not called"
        assert result[0]["filename"] == "x.png"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_detect_scalebar_delegates_to_lib():
    """logic.detect_scalebar must delegate to ZebrafishAnalysisLib.logic.detect_scalebar."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        sentinel = {"bar_found": True, "scale_um_per_px": 22.99}
        with patch("ZebrafishAnalysisLib.logic.detect_scalebar", return_value=sentinel) as mock:
            result = logic.detect_scalebar("/img.png", label_um=500.0)
        assert mock.called
        assert result["scale_um_per_px"] == 22.99
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_preload_models_delegates_to_lib():
    """logic.preload_models must delegate to ZebrafishAnalysisLib.logic.preload_models."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        params = {"curvature": False, "eyes": False, "body_model_filename": "m.pth"}
        with patch("ZebrafishAnalysisLib.logic.preload_models") as mock:
            logic.preload_models(params)
        assert mock.called
        assert mock.call_args[0][0] is params
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_apply_manual_correction_delegates_to_lib():
    """logic.apply_manual_correction must delegate to ZebrafishAnalysisLib.logic."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        import numpy as np

        result = {"mask": np.zeros((256, 256), dtype="uint8"),
                  "original": np.zeros((256, 256, 3), dtype="uint8"),
                  "spacing": (1.0, 1.0)}
        logic = ZebrafishAnalysisLogic()

        sentinel = object()
        with patch("ZebrafishAnalysisLib.logic.apply_manual_correction",
                   return_value=sentinel) as mock:
            ret = logic.apply_manual_correction(result, (10, 20), (200, 220))
        assert mock.called
        assert ret is sentinel
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_revert_manual_correction_delegates_to_lib():
    """logic.revert_manual_correction must delegate to ZebrafishAnalysisLib.logic."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        result = {"manual_corrected": True}
        logic = ZebrafishAnalysisLogic()

        sentinel = object()
        with patch("ZebrafishAnalysisLib.logic.revert_manual_correction",
                   return_value=sentinel) as mock:
            ret = logic.revert_manual_correction(result)
        assert mock.called
        assert mock.call_args[0][0] is result
        assert ret is sentinel
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------

def test_importing_main_module_does_not_import_torch():
    """Importing ZebrafishAnalysis (with stubs) must not pull in torch."""
    r = _run("""
        import sys
        torch_before = "torch" in sys.modules

        from ZebrafishAnalysis import ZebrafishAnalysisLogic  # noqa: F401

        torch_after = "torch" in sys.modules
        assert not torch_after, "torch was imported at module-import time"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_core_not_imported_at_logic_class_definition():
    """ZebrafishAnalysisCore.seg must not be imported when the module loads."""
    r = _run("""
        import sys
        from ZebrafishAnalysis import ZebrafishAnalysisLogic  # noqa: F401

        seg_in_modules = any(
            k.startswith("ZebrafishAnalysisCore.seg")
            for k in sys.modules
        )
        assert not seg_in_modules, "ZebrafishAnalysisCore.seg imported too early"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Input validation — AnalysisInputError contract
# ---------------------------------------------------------------------------

def _check_raises_input_error(code: str) -> None:
    """Helper: assert code raises AnalysisInputError (a ValueError subclass)."""
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert "TYPE:AnalysisInputError" in r.stdout, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_rejects_empty_image_list():
    """run_analysis raises AnalysisInputError for an empty image list."""
    _check_raises_input_error("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        try:
            logic.run_analysis([], {"um_per_px": 1.0})
            print("NO_ERROR")
        except ValueError as exc:
            print(f"TYPE:{type(exc).__name__}")
    """)


def test_logic_run_analysis_rejects_string_image_paths():
    """run_analysis raises AnalysisInputError when image_paths is a plain string."""
    _check_raises_input_error("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        try:
            logic.run_analysis("/single/path.png", {"um_per_px": 1.0})
            print("NO_ERROR")
        except ValueError as exc:
            print(f"TYPE:{type(exc).__name__}")
    """)


def test_logic_run_analysis_rejects_non_sequence_image_paths():
    """run_analysis raises AnalysisInputError when image_paths is not a Sequence.

    The API contract requires a Sequence (list, tuple, …); generators and sets
    are rejected because they do not expose indexing or a deterministic length.
    """
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        bad_inputs = [
            42,                        # not iterable at all
            {"/a.png", "/b.png"},      # set — not a Sequence
            (p for p in ["/a.png"]),   # generator — not a Sequence
        ]
        for val in bad_inputs:
            try:
                logic.run_analysis(val, {"um_per_px": 1.0})
                print(f"NO_ERROR:{type(val).__name__}")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__}:{type(val).__name__}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("TYPE:AnalysisInputError") == 3, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_rejects_non_path_entry():
    """run_analysis raises AnalysisInputError when an entry in image_paths is not path-like."""
    _check_raises_input_error("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        try:
            logic.run_analysis([42, "/valid.png"], {"um_per_px": 1.0})
            print("NO_ERROR")
        except ValueError as exc:
            print(f"TYPE:{type(exc).__name__}")
    """)


def test_logic_run_analysis_rejects_non_mapping_params():
    """run_analysis raises AnalysisInputError when params is not a Mapping."""
    _check_raises_input_error("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        try:
            logic.run_analysis(["/x.png"], "not_a_dict")
            print("NO_ERROR")
        except ValueError as exc:
            print(f"TYPE:{type(exc).__name__}")
    """)


def test_logic_run_analysis_rejects_get_only_non_mapping_params():
    """run_analysis rejects an object that has .get() but is not a collections.abc.Mapping.

    isinstance(x, Mapping) is the correct check, not hasattr(x, 'get').
    """
    _check_raises_input_error("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()

        class FakeGet:
            def get(self, key, default=None):
                return default

        try:
            logic.run_analysis(["/x.png"], FakeGet())
            print("NO_ERROR")
        except ValueError as exc:
            print(f"TYPE:{type(exc).__name__}")
    """)


def test_logic_run_analysis_rejects_um_per_px_below_minimum():
    """run_analysis raises AnalysisInputError for um_per_px below the UI minimum (0.001)."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        for val in (0.0, -1.5, 0.0005):  # 0.0005 is in (0, 0.001) — below UI range
            try:
                logic.run_analysis(["/x.png"], {"um_per_px": val})
                print(f"NO_ERROR:{val}")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__}:{val}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("TYPE:AnalysisInputError") == 3, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_rejects_um_per_px_above_maximum():
    """run_analysis raises AnalysisInputError for um_per_px above the UI maximum (9999.0)."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        for val in (9999.001, 10000.0, 1e9):
            try:
                logic.run_analysis(["/x.png"], {"um_per_px": val})
                print(f"NO_ERROR:{val}")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__}:{val}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("TYPE:AnalysisInputError") == 3, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_rejects_nan_and_inf_for_numeric_params():
    """run_analysis raises AnalysisInputError for NaN or Inf in um_per_px or threshold."""
    r = _run("""
        import math
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()

        cases = [
            {"um_per_px": float("nan")},
            {"um_per_px": float("inf")},
            {"um_per_px": float("-inf")},
            {"um_per_px": 1.0, "threshold": float("nan")},
            {"um_per_px": 1.0, "threshold": float("inf")},
        ]
        for params in cases:
            try:
                logic.run_analysis(["/x.png"], params)
                print(f"NO_ERROR:{params}")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("TYPE:AnalysisInputError") == 5, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_rejects_threshold_out_of_range():
    """run_analysis raises AnalysisInputError for threshold outside [0.0, 1.0]."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisLogic
        logic = ZebrafishAnalysisLogic()
        for val in (-0.01, 1.01, 2.0):
            try:
                logic.run_analysis(["/x.png"], {"um_per_px": 1.0, "threshold": val})
                print(f"NO_ERROR:{val}")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__}:{val}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("TYPE:AnalysisInputError") == 3, r.stdout
    assert "NO_ERROR" not in r.stdout


def test_logic_run_analysis_normalizes_pathlib_path_to_str():
    """run_analysis converts pathlib.Path entries to str before delegating."""
    r = _run("""
        from pathlib import Path
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[]) as mock:
            logic.run_analysis([Path("/tmp/fish.png")], {"um_per_px": 1.0})
        delegated_paths = mock.call_args[0][0]
        assert delegated_paths == ["/tmp/fish.png"], f"expected str path, got {delegated_paths!r}"
        assert all(isinstance(p, str) for p in delegated_paths), "paths must be str after normalization"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_run_analysis_accepts_numeric_string_um_per_px():
    """run_analysis converts a numeric string um_per_px to float and delegates it."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[]) as mock:
            logic.run_analysis(["/x.png"], {"um_per_px": "22.99"})
        delegated_params = mock.call_args[0][1]
        assert delegated_params["um_per_px"] == 22.99, (
            f"expected float 22.99, got {delegated_params['um_per_px']!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_run_analysis_does_not_mutate_caller_image_paths():
    """run_analysis must not modify the original image_paths list."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        original = ["/a.png", "/b.png"]
        snapshot = list(original)
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[]):
            logic.run_analysis(original, {"um_per_px": 1.0})
        assert original == snapshot, f"image_paths mutated: {original!r} != {snapshot!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_run_analysis_does_not_mutate_caller_params():
    """run_analysis must not modify the original params dict."""
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        original = {"um_per_px": "22.99", "threshold": "0.85", "extra": "kept"}
        snapshot = dict(original)
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[]):
            logic.run_analysis(["/x.png"], original)
        assert original == snapshot, f"params mutated: {original!r} != {snapshot!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_logic_run_analysis_passes_progress_callback():
    """run_analysis passes the progress_callback through to analyse_images."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        cb = MagicMock()
        with patch("ZebrafishAnalysisLib.logic.analyse_images", return_value=[]) as mock:
            logic.run_analysis(["/x.png"], {"um_per_px": 1.0}, cb)
        assert mock.call_args[0][2] is cb, f"callback not passed through: {mock.call_args}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_unexpected_core_value_errors_propagate_as_plain_value_error():
    """A ValueError raised inside analyse_images must not be silently retyped.

    AnalysisInputError is raised only in ZebrafishAnalysisLogic validation.
    Unexpected errors from the core propagate as their original type so the
    widget's except-Exception branch (errorDisplay) handles them, not the
    warning branch.
    """
    r = _run("""
        from unittest.mock import patch
        from ZebrafishAnalysis import ZebrafishAnalysisLogic

        logic = ZebrafishAnalysisLogic()
        with patch("ZebrafishAnalysisLib.logic.analyse_images",
                   side_effect=ValueError("unexpected core error")):
            try:
                logic.run_analysis(["/x.png"], {"um_per_px": 1.0})
                print("NO_ERROR")
            except ValueError as exc:
                print(f"TYPE:{type(exc).__name__} MSG:{exc}")
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPE:ValueError " in r.stdout, r.stdout  # plain ValueError, not subclass
    assert "AnalysisInputError" not in r.stdout
    assert "unexpected core error" in r.stdout


# ---------------------------------------------------------------------------
# CMake packaging
# ---------------------------------------------------------------------------

def test_cmake_packaging_list_includes_errors_py():
    """ZebrafishAnalysis/CMakeLists.txt must list errors.py in the SCRIPTS block.

    errors.py is a runtime dependency of widget.py; omitting it would cause
    a ModuleNotFoundError when the packaged extension loads.
    """
    cmake_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ZebrafishAnalysis", "CMakeLists.txt",
    )
    content = open(cmake_path).read()
    assert "ZebrafishAnalysisLib/errors.py" in content, (
        f"CMakeLists.txt does not include ZebrafishAnalysisLib/errors.py:\n{cmake_path}"
    )


# ---------------------------------------------------------------------------
# Widget exception routing
# ---------------------------------------------------------------------------

def test_widget_does_not_call_logic_run_analysis_directly():
    """widget.py must not call self._logic.run_analysis() directly.

    Inference is now subprocess-based (InferenceController / inference_worker.py).
    The main thread never blocks on synchronous model inference.
    """
    path = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib/widget.py")
    source = open(path).read()

    assert "self._logic.run_analysis(" not in source, (
        "widget.py must not call self._logic.run_analysis() directly — "
        "inference runs in a subprocess via InferenceController"
    )


# ---------------------------------------------------------------------------
# UI layer must not import ZebrafishAnalysisLib.logic directly
# ---------------------------------------------------------------------------

def test_widget_and_detail_tab_have_no_direct_logic_lib_imports():
    """widget.py and detail_tab.py must not import ZebrafishAnalysisLib.logic directly.

    Analysis calls must flow through ZebrafishAnalysisLogic, not bypass it.
    Imports inside ZebrafishAnalysisLogic itself (ZebrafishAnalysis.py) are fine.
    """
    import re
    ui_files = ["ZebrafishAnalysisLib/widget.py", "ZebrafishAnalysisLib/detail_tab.py"]
    direct_import_re = re.compile(
        r"from ZebrafishAnalysisLib\.logic import"
        r"|from ZebrafishAnalysisLib import logic"
    )
    violations = []
    for filename in ui_files:
        path = os.path.join(_MODULE_DIR, filename)
        for lineno, line in enumerate(open(path), 1):
            if direct_import_re.search(line):
                violations.append(f"{filename}:{lineno}: {line.rstrip()}")
    assert not violations, (
        "UI files contain direct ZebrafishAnalysisLib.logic imports:\n"
        + "\n".join(violations)
    )
