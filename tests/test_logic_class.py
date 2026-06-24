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
sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=object,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
    VTKObservationMixin=object,
)
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
