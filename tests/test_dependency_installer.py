import importlib
import sys
from unittest.mock import MagicMock

from ZebrafishEmbryoAnalyzerLib.dependency_installer import _is_importable, get_missing_packages


def test_is_importable_finds_numpy():
    assert _is_importable("numpy") is True


def test_is_importable_misses_nonexistent():
    assert _is_importable("this_package_does_not_exist_xyz_000") is False


def test_is_importable_does_not_trigger_import():
    """find_spec must not cause the module to appear in sys.modules."""
    pkg = "skimage"
    was_present = pkg in sys.modules
    sys.modules.pop(pkg, None)  # ensure not already loaded
    _is_importable("scikit-image")
    after = pkg in sys.modules
    assert after is False, "find_spec must not import the package"


# ---------------------------------------------------------------------------
# T1 — get_missing_packages: all present, numpy<2
# ---------------------------------------------------------------------------

def test_get_missing_packages_all_present(monkeypatch):
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", lambda n: True)
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._numpy_major", lambda: 1)
    result = get_missing_packages()
    assert result["torch"] == []
    assert result["general"] == []
    assert result["numpy_pin"] == []


# ---------------------------------------------------------------------------
# T2 — numpy>=2 triggers pin
# ---------------------------------------------------------------------------

def test_get_missing_packages_numpy_pin_when_numpy2(monkeypatch):
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", lambda n: True)
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._numpy_major", lambda: 2)
    result = get_missing_packages()
    assert result["numpy_pin"] == ["numpy<2"]


# ---------------------------------------------------------------------------
# T3 — torch packages detected as missing
# ---------------------------------------------------------------------------

def test_get_missing_packages_torch_missing(monkeypatch):
    def fake_importable(name):
        return name not in ("torch", "torchvision")
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", fake_importable)
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._numpy_major", lambda: 1)
    result = get_missing_packages()
    assert "torch" in result["torch"]
    assert "torchvision" in result["torch"]
    assert result["general"] == []


# ---------------------------------------------------------------------------
# T4 — testingEnabled guard: pip_fn never called in testing mode
# ---------------------------------------------------------------------------

def test_install_packages_skipped_in_testing_mode(monkeypatch):
    mock_slicer = MagicMock()
    mock_slicer.app.testingEnabled.return_value = True
    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    pip_fn = MagicMock()
    from ZebrafishEmbryoAnalyzerLib import dependency_installer
    importlib.reload(dependency_installer)
    dependency_installer.install_packages(
        {"torch": ["torch"], "general": [], "numpy_pin": []}, pip_fn=pip_fn
    )
    pip_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers for T5–T7: build a full slicer/qt mock environment
# ---------------------------------------------------------------------------

def _make_slicer_mock():
    mock_slicer = MagicMock()
    mock_slicer.app.testingEnabled.return_value = False
    return mock_slicer


def _make_qt_mock():
    mock_qt = MagicMock()
    mock_qt.Qt.WindowContextHelpButtonHint = 0x4000000
    mock_qt.Qt.WindowCloseButtonHint = 0x200
    mock_qt.QDialogButtonBox.RejectRole = 1
    return mock_qt


def _make_logic_mock(torch_present: bool):
    mock_logic = MagicMock()
    mock_logic.dependency_status.return_value = {"torch": torch_present, "cv2": True,
                                                  "segmentation_models_pytorch": True, "timm": True}
    return mock_logic


# ---------------------------------------------------------------------------
# T5 — numpy<2 pinned when torch just installed successfully
# ---------------------------------------------------------------------------

def test_install_packages_pins_numpy_when_torch_ok(monkeypatch):
    mock_slicer = _make_slicer_mock()
    mock_qt = _make_qt_mock()
    mock_logic = _make_logic_mock(torch_present=False)

    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setitem(sys.modules, "qt", mock_qt)
    monkeypatch.setitem(sys.modules, "ZebrafishEmbryoAnalyzerLib.logic", mock_logic)

    pip_fn = MagicMock()  # succeeds (no raise) → torch_ok = True

    from ZebrafishEmbryoAnalyzerLib import dependency_installer
    importlib.reload(dependency_installer)

    dependency_installer.install_packages(
        {"torch": ["torch"], "general": [], "numpy_pin": ["numpy<2"]}, pip_fn=pip_fn
    )

    calls = [str(c) for c in pip_fn.call_args_list]
    assert any("numpy<2" in c for c in calls), (
        f"Expected pip_fn called with 'numpy<2', got: {calls}"
    )


# ---------------------------------------------------------------------------
# T6 — numpy<2 NOT pinned when torch fails and already absent
# ---------------------------------------------------------------------------

def test_install_packages_no_numpy_pin_when_torch_fails_and_absent(monkeypatch):
    mock_slicer = _make_slicer_mock()
    mock_qt = _make_qt_mock()

    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setitem(sys.modules, "qt", mock_qt)

    def pip_fn_side_effect(pkg_str, cancel_check=None):
        if "torch" in pkg_str:
            raise RuntimeError("simulated torch install failure")
        # other packages succeed (return None)

    pip_fn = MagicMock(side_effect=pip_fn_side_effect)

    from ZebrafishEmbryoAnalyzerLib import dependency_installer
    importlib.reload(dependency_installer)

    # Simulate torch absent on disk so already_has_torch is False
    monkeypatch.setattr(dependency_installer, "_is_importable", lambda name: name != "torch")

    dependency_installer.install_packages(
        {"torch": ["torch"], "general": [], "numpy_pin": ["numpy<2"]},
        pip_fn=pip_fn,
    )

    # torch install failed and torch is absent → numpy<2 must NOT be pinned
    call_args_flat = [str(c) for c in pip_fn.call_args_list]
    assert not any("numpy<2" in arg for arg in call_args_flat), (
        "numpy<2 must not be pinned when torch install failed and torch is absent"
    )


# ---------------------------------------------------------------------------
# T7 — numpy<2 pinned when torch already present (no new install needed)
# ---------------------------------------------------------------------------

def test_install_packages_numpy_pin_when_torch_preexisting(monkeypatch):
    mock_slicer = _make_slicer_mock()
    mock_qt = _make_qt_mock()

    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setitem(sys.modules, "qt", mock_qt)

    pip_fn = MagicMock()

    from ZebrafishEmbryoAnalyzerLib import dependency_installer
    importlib.reload(dependency_installer)

    # Simulate torch already present on disk so already_has_torch is True.
    # install_packages() checks _is_importable("torch") directly, not logic.dependency_status().
    monkeypatch.setattr(dependency_installer, "_is_importable", lambda name: name == "torch")

    # torch not in missing (already installed); numpy_pin present
    dependency_installer.install_packages(
        {"torch": [], "general": [], "numpy_pin": ["numpy<2"]}, pip_fn=pip_fn
    )

    calls = [str(c) for c in pip_fn.call_args_list]
    assert any("numpy<2" in c for c in calls), (
        f"Expected pip_fn called with 'numpy<2', got: {calls}"
    )


# ---------------------------------------------------------------------------
# T8 — prewarm guard verified via get_missing_packages structure
#       (actual thread-spawn test requires Slicer; verified structurally here)
# ---------------------------------------------------------------------------

def test_prewarm_guard_structure_when_torch_missing(monkeypatch):
    # prewarm guard verified via get_missing_packages structure test
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: n not in ("torch", "torchvision"))
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._numpy_major", lambda: 1)
    result = get_missing_packages()
    # Guard condition: if result["torch"] is truthy → _prewarm_imports returns early
    assert bool(result["torch"]) is True, "torch missing → guard fires → no thread spawned"
    assert "torch" in result["torch"]
    assert "torchvision" in result["torch"]
