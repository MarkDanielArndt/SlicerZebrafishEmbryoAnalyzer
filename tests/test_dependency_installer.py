import importlib
import sys
from unittest.mock import MagicMock

import pytest

from ZebrafishEmbryoAnalyzerLib.dependency_installer import _is_importable, get_missing_packages


def test_is_importable_finds_numpy():
    assert _is_importable("numpy") is True


def test_is_importable_misses_nonexistent():
    assert _is_importable("this_package_does_not_exist_xyz_000") is False


def test_is_importable_does_not_trigger_import():
    """find_spec must not cause the module to appear in sys.modules."""
    pkg = "skimage"
    sys.modules.pop(pkg, None)  # ensure not already loaded
    _is_importable("scikit-image")
    assert pkg not in sys.modules, "find_spec must not import the package"


# ---------------------------------------------------------------------------
# get_missing_packages
# ---------------------------------------------------------------------------

def test_get_missing_packages_all_present(monkeypatch):
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", lambda n: True)
    result = get_missing_packages()
    assert result["torch"] == []
    assert result["general"] == []


def test_get_missing_packages_torch_missing(monkeypatch):
    def fake_importable(name):
        return name not in ("torch", "torchvision")
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", fake_importable)
    result = get_missing_packages()
    assert result["torch"] == ["torch", "torchvision"]
    assert result["general"] == []


def test_get_missing_packages_reports_no_numpy_pin(monkeypatch):
    """The global numpy<2 pin is gone — it changed the environment for every
    other extension in the same Slicer installation."""
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable", lambda n: True)
    assert "numpy_pin" not in get_missing_packages()


def test_excel_export_does_not_pull_in_torch(monkeypatch):
    """Exporting a result table must not trigger a two-gigabyte torch install."""
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: False)
    result = get_missing_packages("excel")
    assert result["torch"] == []
    assert result["general"] == ["openpyxl"]


def test_loading_images_needs_opencv_but_not_torch(monkeypatch):
    """Loading images reads every file with cv2.imread and builds thumbnails with
    cv2.resize, so OpenCV is required before the gallery can show anything — but nothing
    from the ML stack is."""
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: False)
    result = get_missing_packages("images")
    assert result["torch"] == []
    assert result["general"] == ["opencv-python-headless"]


def test_scalebar_needs_only_vision_packages(monkeypatch):
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: False)
    result = get_missing_packages("scalebar")
    assert result["torch"] == []
    assert set(result["general"]) == {"opencv-python-headless", "pytesseract"}


def test_analysis_needs_everything(monkeypatch):
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: False)
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    result = get_missing_packages("analysis")
    assert result["torch"] == di.TORCH_PACKAGES
    assert result["general"] == di.REQUIRED_PACKAGES


def test_unknown_purpose_falls_back_to_everything(monkeypatch):
    """A typo in a call site must not silently skip the install."""
    monkeypatch.setattr("ZebrafishEmbryoAnalyzerLib.dependency_installer._is_importable",
                        lambda n: False)
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    assert get_missing_packages("typo") == get_missing_packages("analysis")
    assert get_missing_packages("typo")["torch"] == di.TORCH_PACKAGES


def test_get_missing_packages_imports_no_slicer():
    """Must stay pure Python so it is safe to call at any site, including setup()."""
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    source = open(di.__file__).read()
    header = source.split("def _pytorch_utils_logic")[0]
    assert "import slicer" not in header
    assert "import qt" not in header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_with_slicer(monkeypatch, testing_enabled=False, platform="linux", numpy_major=2):
    """Install a slicer mock and reload the module against it. Returns (module, mock).

    platform and numpy_major are pinned explicitly: the numpy constraint is macOS-only,
    so tests must not inherit the developer machine's platform or numpy version.
    """
    mock_slicer = MagicMock()
    mock_slicer.app.testingEnabled.return_value = testing_enabled
    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setattr(sys, "platform", platform)
    from ZebrafishEmbryoAnalyzerLib import dependency_installer
    importlib.reload(dependency_installer)
    monkeypatch.setattr(dependency_installer, "_numpy_major", lambda: numpy_major)
    return dependency_installer, mock_slicer


# ---------------------------------------------------------------------------
# install_packages
# ---------------------------------------------------------------------------

def test_install_packages_skipped_in_testing_mode(monkeypatch):
    di, _ = _reload_with_slicer(monkeypatch, testing_enabled=True)
    pip_fn, torch_fn = MagicMock(), MagicMock()
    assert di.install_packages({"torch": ["torch"], "general": ["timm"]},
                               pip_fn=pip_fn, torch_fn=torch_fn) == "skipped"
    pip_fn.assert_not_called()
    torch_fn.assert_not_called()


def test_install_packages_never_pip_installs_torch(monkeypatch):
    """Torch must go through the PyTorch extension, never through pip."""
    di, _ = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock()
    torch_fn = MagicMock(return_value="ok")

    di.install_packages({"torch": ["torch", "torchvision"], "general": ["timm"]},
                        pip_fn=pip_fn, torch_fn=torch_fn)

    torch_fn.assert_called_once()
    pip_fn.assert_called_once_with("timm")
    assert "torch" not in pip_fn.call_args.args[0]


def test_install_packages_uses_a_single_pip_invocation(monkeypatch):
    """All packages go into one pip call so the resolver sees the whole set at once.

    Installing them one by one lets a later package pull in a dependency that breaks an
    earlier one — scikit-image drags in tifffile, which requires numpy>=2.1.
    """
    di, _ = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock()
    di.install_packages({"torch": [], "general": ["timm", "scikit-image", "openpyxl"]},
                        pip_fn=pip_fn, torch_fn=MagicMock())
    pip_fn.assert_called_once_with("timm scikit-image openpyxl")


def test_install_reports_ready_when_numpy_was_not_replaced(monkeypatch):
    """The common case: nothing already held in memory changed, so the caller can just
    carry on without restarting Slicer."""
    di, _ = _reload_with_slicer(monkeypatch, platform="linux")
    monkeypatch.setattr(di, "_numpy_version", lambda: "2.4.6")

    assert di.install_packages({"torch": [], "general": ["timm"]},
                               pip_fn=MagicMock(), torch_fn=MagicMock()) == "ready"


def test_install_reports_restart_when_numpy_version_changed(monkeypatch):
    """Installing torch on macOS downgrades numpy, which Slicer has already imported —
    that is the case where a restart is genuinely unavoidable."""
    di, _ = _reload_with_slicer(monkeypatch, platform="darwin")
    versions = iter(["2.4.6", "1.26.4"])
    monkeypatch.setattr(di, "_numpy_version", lambda: next(versions))

    assert di.install_packages({"torch": [], "general": ["scikit-image"]},
                               pip_fn=MagicMock(), torch_fn=MagicMock()) == "restart"


def test_pip_failure_is_left_to_slicers_own_dialog(monkeypatch):
    """slicer.util.pip_install already shows an error dialog containing the full pip log
    before it raises. A second dialog from us would only add the exception text, which is
    just "non-zero exit status"."""
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock(side_effect=RuntimeError("simulated failure"))

    assert di.install_packages({"torch": [], "general": ["timm", "openpyxl"]},
                               pip_fn=pip_fn, torch_fn=MagicMock()) == "failed"

    mock_slicer.util.errorDisplay.assert_not_called()


def test_install_packages_aborts_remaining_when_torch_fails(monkeypatch):
    """A failed torch install must stop the run — otherwise pip resolves torch as a
    transitive dependency of the remaining packages and bypasses the PyTorch extension."""
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock()
    torch_fn = MagicMock(side_effect=RuntimeError("no extension server"))

    assert di.install_packages({"torch": ["torch"], "general": ["segmentation_models_pytorch"]},
                               pip_fn=pip_fn, torch_fn=torch_fn) == "failed"

    pip_fn.assert_not_called()
    mock_slicer.util.errorDisplay.assert_called_once()
    assert "PyTorch" in mock_slicer.util.errorDisplay.call_args.args[0]


def test_install_packages_aborts_remaining_when_extension_needs_restart(monkeypatch):
    """After installing the PyTorch extension, torch itself is not available until Slicer
    restarts. Installing the remaining packages now would let pip pull its own torch."""
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock()

    assert di.install_packages({"torch": ["torch"], "general": ["segmentation_models_pytorch"]},
                               pip_fn=pip_fn, torch_fn=MagicMock(return_value="restart")) == "restart"

    pip_fn.assert_not_called()
    mock_slicer.util.errorDisplay.assert_not_called()
    mock_slicer.util.infoDisplay.assert_called_once()
    assert "estart" in mock_slicer.util.infoDisplay.call_args.args[0]


def test_install_packages_proceeds_after_torch_ok(monkeypatch):
    """Only once torch is actually installed may the remaining packages follow."""
    di, _ = _reload_with_slicer(monkeypatch)
    pip_fn = MagicMock()

    assert di.install_packages({"torch": ["torch"], "general": ["timm"]},
                               pip_fn=pip_fn, torch_fn=MagicMock(return_value="ok")) == "ready"
    pip_fn.assert_called_once_with("timm")


def test_install_packages_defaults_to_slicer_pip_install(monkeypatch):
    """Without injection, the default must be slicer.util.pip_install — not a
    custom subprocess call."""
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    di.install_packages({"torch": [], "general": ["timm"]}, torch_fn=MagicMock())
    mock_slicer.util.pip_install.assert_called_once_with("timm")


def test_no_custom_pip_helper_remains():
    """_pip_install and the hard-coded CPU wheel index are gone."""
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    assert not hasattr(di, "_pip_install")
    assert not hasattr(di, "TORCH_INDEX")
    assert "download.pytorch.org" not in open(di.__file__).read()


# ---------------------------------------------------------------------------
# PyTorch extension handling
# ---------------------------------------------------------------------------

def test_install_torch_requests_extension_when_missing(monkeypatch):
    """Without PyTorchUtils, the extension is installed and a restart signalled."""
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    monkeypatch.setattr(di, "_pytorch_utils_logic", lambda: None)
    manager = mock_slicer.app.extensionsManagerModel.return_value
    manager.isExtensionInstalled.return_value = False
    manager.installExtensionFromServer.return_value = True

    assert di._install_torch() == "restart"
    manager.installExtensionFromServer.assert_called_once_with("PyTorch")


def test_install_torch_raises_when_extension_unavailable(monkeypatch):
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    monkeypatch.setattr(di, "_pytorch_utils_logic", lambda: None)
    manager = mock_slicer.app.extensionsManagerModel.return_value
    manager.isExtensionInstalled.return_value = False
    manager.installExtensionFromServer.return_value = False

    with pytest.raises(RuntimeError, match="Extensions Manager"):
        di._install_torch()


def test_install_torch_uses_pytorch_utils_when_available(monkeypatch):
    di, _ = _reload_with_slicer(monkeypatch)
    torch_logic = MagicMock()
    torch_logic.installTorch.return_value = object()
    monkeypatch.setattr(di, "_pytorch_utils_logic", lambda: torch_logic)

    assert di._install_torch() == "ok"
    torch_logic.installTorch.assert_called_once_with(askConfirmation=False)


def test_install_torch_raises_when_pytorch_utils_fails(monkeypatch):
    di, _ = _reload_with_slicer(monkeypatch)
    torch_logic = MagicMock()
    torch_logic.installTorch.return_value = None  # PyTorchUtils failure contract
    monkeypatch.setattr(di, "_pytorch_utils_logic", lambda: torch_logic)

    with pytest.raises(RuntimeError, match="PyTorch"):
        di._install_torch()


def test_pytorch_utils_logic_returns_none_without_extension(monkeypatch):
    di, _ = _reload_with_slicer(monkeypatch)
    monkeypatch.setitem(sys.modules, "PyTorchUtils", None)  # forces ModuleNotFoundError
    assert di._pytorch_utils_logic() is None


# ---------------------------------------------------------------------------
# numpy constraint (macOS only) and the opencv version that follows from it
# ---------------------------------------------------------------------------

def test_numpy_major_does_not_import_numpy(monkeypatch):
    """Reading the version must not import numpy — an imported numpy cannot be
    replaced without restarting Slicer."""
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    src = open(di.__file__).read()
    body = src.split("def _numpy_version")[1].split("def ")[0]
    assert "import numpy" not in body.replace("importlib", "")
    assert "importlib.metadata" in body
    assert di._numpy_major() >= 1  # still returns something usable


def test_numpy_not_constrained_off_macos(monkeypatch):
    """On Linux and Windows a current torch supports NumPy 2 — capping it there would
    change Slicer's shared environment for every other extension for no reason."""
    di, _ = _reload_with_slicer(monkeypatch, platform="linux", numpy_major=2)
    pip_fn = MagicMock()

    di.install_packages({"torch": [], "general": ["scikit-image"]},
                        pip_fn=pip_fn, torch_fn=MagicMock(return_value="ok"))

    pip_fn.assert_called_once_with("scikit-image")


def test_numpy_capped_in_the_same_invocation_on_macos(monkeypatch):
    """The cap must travel in the same pip call as the packages, so the resolver picks
    versions of their transitive dependencies that fit numpy 1."""
    di, _ = _reload_with_slicer(monkeypatch, platform="darwin", numpy_major=2)
    pip_fn = MagicMock()

    di.install_packages({"torch": [], "general": ["scikit-image", "opencv-python-headless"]},
                        pip_fn=pip_fn, torch_fn=MagicMock(return_value="ok"))

    pip_fn.assert_called_once_with("scikit-image opencv-python-headless numpy<2")


def test_numpy_capped_on_macos_even_with_nothing_else_to_install(monkeypatch):
    """A numpy that has drifted back above 2 must still be corrected."""
    di, _ = _reload_with_slicer(monkeypatch, platform="darwin", numpy_major=2)
    pip_fn = MagicMock()

    di.install_packages({"torch": [], "general": []},
                        pip_fn=pip_fn, torch_fn=MagicMock(return_value="ok"))

    pip_fn.assert_called_once_with("numpy<2")


def test_nothing_installed_when_nothing_missing_and_numpy_is_fine(monkeypatch):
    di, _ = _reload_with_slicer(monkeypatch, platform="darwin", numpy_major=1)
    pip_fn = MagicMock()

    di.install_packages({"torch": [], "general": []},
                        pip_fn=pip_fn, torch_fn=MagicMock(return_value="ok"))

    pip_fn.assert_not_called()


def test_no_hand_pinned_transitive_dependencies_remain():
    """Pinning individual transitive packages does not scale — opencv was the first case,
    tifffile the second. The cap plus a single resolver pass replaces that approach."""
    import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
    src = open(di.__file__).read()
    assert "opencv-python-headless<" not in src
    assert not hasattr(di, "OPENCV_NUMPY1_SPEC")


def test_install_pytorch_extension_skips_when_already_installed(monkeypatch):
    di, mock_slicer = _reload_with_slicer(monkeypatch)
    manager = mock_slicer.app.extensionsManagerModel.return_value
    manager.isExtensionInstalled.return_value = True

    assert di.install_pytorch_extension() is True
    manager.installExtensionFromServer.assert_not_called()
