"""
F1 regression tests: deferred ML imports.

Verifies that importing ZebrafishEmbryoAnalyzerCore and ZebrafishEmbryoAnalyzerLib.logic
at module level does NOT trigger torch, segmentation_models_pytorch, or cv2.
Also verifies that dependency_status() correctly reports package availability
without importing them.
"""
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
)


def _subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# AC3: Core importable without torch present
# ---------------------------------------------------------------------------

def test_core_seg_import_does_not_trigger_torch():
    """Importing ZebrafishEmbryoAnalyzerCore.seg must not load torch at module level."""
    result = _subprocess(
        """
        import sys
        # Block torch so any module-level import raises ImportError
        sys.modules["torch"] = None
        # Block ML-only deps too
        sys.modules["segmentation_models_pytorch"] = None
        sys.modules["huggingface_hub"] = None
        sys.modules["timm"] = None
        from ZebrafishEmbryoAnalyzerCore import seg
        assert "torch" not in sys.modules or sys.modules["torch"] is None, (
            "torch was imported at module level in seg.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerCore.seg must be importable without torch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_core_length_import_does_not_trigger_torch():
    """Importing ZebrafishEmbryoAnalyzerCore.length must not load torch at module level."""
    result = _subprocess(
        """
        import sys
        sys.modules["torch"] = None
        sys.modules["timm"] = None
        sys.modules["huggingface_hub"] = None
        from ZebrafishEmbryoAnalyzerCore import length
        assert "torch" not in sys.modules or sys.modules["torch"] is None, (
            "torch was imported at module level in length.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerCore.length must be importable without torch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_core_both_modules_import_without_torch():
    """Both core modules importable together without torch."""
    result = _subprocess(
        """
        import sys
        sys.modules["torch"] = None
        sys.modules["segmentation_models_pytorch"] = None
        sys.modules["timm"] = None
        sys.modules["huggingface_hub"] = None
        from ZebrafishEmbryoAnalyzerCore import seg, length
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "Both core modules must be importable without torch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# AC1/AC2: logic.py importable without cv2 or torch at module level
# ---------------------------------------------------------------------------

def test_logic_import_does_not_trigger_cv2():
    """Importing ZebrafishEmbryoAnalyzerLib.logic must not load cv2 at module level."""
    result = _subprocess(
        """
        import sys
        sys.modules["cv2"] = None
        from ZebrafishEmbryoAnalyzerLib import logic
        assert "cv2" not in sys.modules or sys.modules["cv2"] is None, (
            "cv2 was imported at module level in logic.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerLib.logic must be importable without cv2.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_logic_import_does_not_trigger_torch():
    """Importing ZebrafishEmbryoAnalyzerLib.logic must not load torch at module level."""
    result = _subprocess(
        """
        import sys
        sys.modules["torch"] = None
        from ZebrafishEmbryoAnalyzerLib import logic
        assert "torch" not in sys.modules or sys.modules["torch"] is None, (
            "torch was imported at module level in logic.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerLib.logic must be importable without torch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# AC6: dependency_status() reports correct status without importing packages
# ---------------------------------------------------------------------------

def test_dependency_status_returns_dict_with_expected_keys():
    """dependency_status() returns a dict containing required package keys."""
    from ZebrafishEmbryoAnalyzerLib.logic import dependency_status
    status = dependency_status()
    assert isinstance(status, dict), "dependency_status() must return a dict"
    for key in ("torch", "cv2"):
        assert key in status, f"'{key}' missing from dependency_status() result"
    for key, val in status.items():
        assert isinstance(val, bool), f"status[{key!r}] must be bool, got {type(val)}"


def test_dependency_status_does_not_import_packages():
    """dependency_status() must not cause torch or cv2 to be imported."""
    result = _subprocess(
        """
        import sys
        # Block the packages so any import would raise
        sys.modules["torch"] = None
        sys.modules["cv2"] = None
        sys.modules["segmentation_models_pytorch"] = None
        sys.modules["timm"] = None
        from ZebrafishEmbryoAnalyzerLib.logic import dependency_status
        status = dependency_status()
        assert isinstance(status, dict)
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "dependency_status() must not import torch or cv2.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_dependency_status_detects_available_packages():
    """When torch is installed, dependency_status() reports it as available."""
    torch_available = importlib.util.find_spec("torch") is not None
    if not torch_available:
        pytest.skip("torch not installed in this environment")

    from ZebrafishEmbryoAnalyzerLib.logic import dependency_status
    status = dependency_status()
    assert status["torch"] is True, (
        "torch is installed but dependency_status() reports it as unavailable"
    )


def test_dependency_status_reports_missing_package():
    """dependency_status() reports False for a package that does not exist."""
    result = _subprocess(
        """
        import sys
        # Pretend a package does not exist by removing it from sys.modules
        # and blocking its spec. The simplest way: override find_spec.
        import importlib.util as _iu
        _real_find_spec = _iu.find_spec

        def _patched_find_spec(name, *args, **kwargs):
            if name == "torch":
                return None
            return _real_find_spec(name, *args, **kwargs)

        _iu.find_spec = _patched_find_spec
        from ZebrafishEmbryoAnalyzerLib.logic import dependency_status
        status = dependency_status()
        assert status["torch"] is False, f"Expected False for torch, got {status['torch']!r}"
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "dependency_status() must return False for unavailable package.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Lazy import: torch only enters sys.modules when ML function is called
# ---------------------------------------------------------------------------

def test_core_seg_import_does_not_trigger_cv2():
    """Importing ZebrafishEmbryoAnalyzerCore.seg must not load cv2 at module level."""
    result = _subprocess(
        """
        import sys
        sys.modules["cv2"] = None
        sys.modules["torch"] = None
        sys.modules["segmentation_models_pytorch"] = None
        sys.modules["huggingface_hub"] = None
        sys.modules["timm"] = None
        from ZebrafishEmbryoAnalyzerCore import seg
        assert "cv2" not in sys.modules or sys.modules["cv2"] is None, (
            "cv2 was imported at module level in seg.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerCore.seg must be importable without cv2.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_core_length_import_does_not_trigger_cv2():
    """Importing ZebrafishEmbryoAnalyzerCore.length must not load cv2 at module level."""
    result = _subprocess(
        """
        import sys
        sys.modules["cv2"] = None
        from ZebrafishEmbryoAnalyzerCore import length
        assert "cv2" not in sys.modules or sys.modules["cv2"] is None, (
            "cv2 was imported at module level in length.py"
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "ZebrafishEmbryoAnalyzerCore.length must be importable without cv2.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_torch_not_in_sys_modules_after_core_import():
    """torch must not appear in sys.modules after a plain Core import."""
    result = _subprocess(
        """
        import sys
        # Do NOT block torch — just check it doesn't get pulled in passively
        # Remove it first so we start clean
        sys.modules.pop("torch", None)
        from ZebrafishEmbryoAnalyzerCore import seg, length
        if "torch" in sys.modules:
            print("FAIL: torch in sys.modules after core import")
            sys.exit(1)
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "torch must not be in sys.modules after plain core import.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
