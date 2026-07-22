"""
Dependency status check and install helpers for ZebrafishEmbryoAnalyzer.

get_missing_packages()  -- pure Python, safe to call anywhere
install_packages()      -- Slicer-only, called only from explicit user action

Torch is never pip-installed from here. It comes from the PyTorch extension via
PyTorchUtils, which selects the build matching the user's hardware; installing it
ourselves from a fixed wheel index forced a CPU-only build on every platform.
"""

REQUIRED_PACKAGES = [
    "segmentation_models_pytorch",
    "timm",
    "scikit-image",
    "opencv-python-headless",
    "huggingface_hub",
    "openpyxl",
    "pytesseract",
]

# Detection only — provided by the PyTorch extension, never installed via pip here.
TORCH_PACKAGES = ["torch", "torchvision"]

PYTORCH_EXTENSION_NAME = "PyTorch"

# On macOS the newest torch build available is 2.2, which is compiled against the NumPy 1
# C API. Anything resolved alongside it has to fit that, so numpy is capped in the same pip
# invocation and pip picks matching versions of the rest by itself.
NUMPY_TORCH_SPEC = "numpy<2"


def _is_importable(name: str) -> bool:
    import importlib.util
    import_name = {
        "scikit-image":                "skimage",
        "opencv-python-headless":      "cv2",
        "huggingface_hub":             "huggingface_hub",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
    }.get(name, name)
    return importlib.util.find_spec(import_name) is not None


# What each user action actually needs. Exporting a result table should not pull in a
# two-gigabyte torch install, and viewing results needs nothing at all.
# None means "everything", including torch.
PACKAGES_FOR_PURPOSE = {
    "analysis": None,
    # Showing images at all needs OpenCV: the gallery renders every thumbnail through
    # ZebrafishEmbryoAnalyzerLib.overlay, which imports cv2 at module level — even for
    # placeholders of images that have not been analysed yet.
    "images":   ["opencv-python-headless"],
    "scalebar": ["opencv-python-headless", "pytesseract"],
    "excel":    ["openpyxl"],
}


def get_missing_packages(purpose: str = "analysis") -> dict:
    """
    Return {"torch": [...], "general": [...]} for the given purpose.
    Unknown purposes fall back to the full set.
    Pure Python — no slicer/qt import. Safe at any call site.
    """
    wanted = PACKAGES_FOR_PURPOSE.get(purpose, None)
    torch_packages   = TORCH_PACKAGES if wanted is None else []
    general_packages = REQUIRED_PACKAGES if wanted is None else [
        p for p in REQUIRED_PACKAGES if p in wanted
    ]
    return {
        "torch":   [p for p in torch_packages   if not _is_importable(p)],
        "general": [p for p in general_packages if not _is_importable(p)],
    }


def _numpy_version() -> str:
    """Installed numpy version as reported by package metadata, or "" if unknown.

    Reads metadata instead of importing numpy on purpose: once numpy has been imported
    it can no longer be replaced without restarting Slicer, which would defeat the
    constraint applied alongside the torch install.
    """
    import importlib.metadata
    try:
        return importlib.metadata.version("numpy")
    except Exception:
        return ""


def _numpy_major() -> int:
    """Installed numpy major version, or 0 if it cannot be determined."""
    try:
        return int(_numpy_version().split(".")[0])
    except Exception:
        return 0


def _numpy_must_stay_v1() -> bool:
    """True where the available torch build is compiled against the NumPy 1 C API.

    Only macOS: torch 2.2 is the last release with a macOS wheel, and against NumPy 2 it
    imports with just a warning and then fails at the first array conversion with "Numpy is
    not available". Slicer's own PyTorch extension applies the same constraint during its
    install. Not done on other platforms, where a current torch supports NumPy 2 and
    downgrading would change Slicer's shared environment for every other extension for
    no reason.
    """
    import sys
    return sys.platform == "darwin"


def _pytorch_utils_logic():
    """PyTorchUtils logic from the PyTorch extension, or None if it is not installed."""
    try:
        import PyTorchUtils
    except ModuleNotFoundError:
        return None
    return PyTorchUtils.PyTorchUtilsLogic()


def install_pytorch_extension() -> bool:
    """Install the PyTorch extension from the extension server.

    Returns True once it is installed. Slicer must be restarted before
    PyTorchUtils becomes importable.
    """
    import slicer
    manager = slicer.app.extensionsManagerModel()
    if manager.isExtensionInstalled(PYTORCH_EXTENSION_NAME):
        return True
    return bool(manager.installExtensionFromServer(PYTORCH_EXTENSION_NAME))


def _install_torch() -> str:
    """Install torch and torchvision through the PyTorch extension.

    Returns "ok" when torch is installed, or "restart" when the PyTorch extension
    itself had to be installed first — PyTorchUtils only becomes importable after
    a Slicer restart, so torch follows on the next run.
    Raises RuntimeError if neither can be installed.
    """
    torch_logic = _pytorch_utils_logic()
    if torch_logic is None:
        if not install_pytorch_extension():
            raise RuntimeError(
                "The PyTorch extension could not be installed from the extension server. "
                "Install it manually via the Extensions Manager and restart Slicer."
            )
        return "restart"

    if torch_logic.installTorch(askConfirmation=False) is None:
        raise RuntimeError("PyTorch could not be installed through the PyTorch extension.")
    return "ok"


def install_packages(missing: dict, pip_fn=None, torch_fn=None) -> str:
    """
    Install missing packages. Only called from an explicit user action.

    missing:  dict from get_missing_packages()
    pip_fn:   injectable for testing (default: slicer.util.pip_install)
    torch_fn: injectable for testing (default: _install_torch)

    Returns one of:
      "skipped"  nothing was done (testing mode)
      "failed"   the install did not succeed; the user has been told
      "restart"  installed, but Slicer must restart before the packages can be used
      "ready"    installed and usable immediately — the caller may just continue

    "ready" is the common case: a freshly pip-installed package that this session never
    imported is importable straight away. A restart is only needed when something already
    held in memory was replaced, which in practice means numpy — installing torch on macOS
    downgrades it.
    """
    import slicer
    if slicer.app.testingEnabled():
        return "skipped"

    import logging

    if pip_fn is None:
        pip_fn = slicer.util.pip_install
    if torch_fn is None:
        torch_fn = _install_torch

    # Torch must be settled before anything else is installed. Several of the remaining
    # packages (segmentation_models_pytorch above all) declare torch as a dependency, so
    # pip would happily resolve and install its own torch build if we got here first —
    # bypassing the PyTorch extension and the platform-specific constraints it applies.
    # On macOS that produced a torch compiled against NumPy 1.x sitting next to NumPy 2:
    # it imports, then fails at the first array conversion with "Numpy is not available".
    if missing.get("torch"):
        slicer.util.showStatusMessage("ZebrafishEmbryoAnalyzer: installing PyTorch…")
        try:
            outcome = torch_fn()
        except Exception as exc:
            logging.exception("Failed to install PyTorch: %s", exc)
            slicer.util.errorDisplay(
                f"PyTorch could not be installed:\n\n{exc}\n\n"
                "No further packages were installed."
            )
            return "failed"

        if outcome == "restart":
            slicer.util.infoDisplay(
                "The PyTorch extension has been installed.\n\n"
                "Restart Slicer and open this module again to install the remaining "
                "packages."
            )
            return "restart"

    # One pip invocation for everything. Installing package by package lets pip resolve each
    # one in isolation, so a later package can pull in a dependency that breaks an earlier
    # one: scikit-image drags in tifffile, whose current release requires numpy>=2.1, which
    # replaced the numpy 1.x that the macOS torch build needs. Handing pip the whole set at
    # once lets it pick versions that fit together — including older tifffile and opencv
    # releases — instead of us chasing each transitive dependency by hand.
    requirements = list(missing.get("general", []))
    if _numpy_must_stay_v1() and (requirements or _numpy_major() >= 2):
        requirements.append(NUMPY_TORCH_SPEC)

    numpy_before = _numpy_version()

    if requirements:
        slicer.util.showStatusMessage(
            f"ZebrafishEmbryoAnalyzer: installing {len(requirements)} Python packages…"
        )
        try:
            pip_fn(" ".join(requirements))
        except Exception as exc:
            # No dialog here: slicer.util.pip_install already showed one containing the
            # full pip log before raising. Ours would be a second, worse dialog — the
            # exception text alone is just "non-zero exit status".
            logging.exception("Failed to install Python packages: %s", exc)
            return "failed"

    # numpy is the one package Slicer has already imported by the time we get here, so a
    # changed version is what actually forces a restart. Everything else was missing a
    # moment ago and can therefore be imported straight away.
    if _numpy_version() != numpy_before:
        slicer.util.showStatusMessage(
            "ZebrafishEmbryoAnalyzer: dependencies installed — restart required."
        )
        return "restart"

    slicer.util.showStatusMessage("ZebrafishEmbryoAnalyzer: dependencies installed.")
    return "ready"
