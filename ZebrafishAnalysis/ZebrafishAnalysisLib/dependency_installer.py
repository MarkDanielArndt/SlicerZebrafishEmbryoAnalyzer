"""
Dependency status check and install helpers for ZebrafishAnalysis.

get_missing_packages()  -- pure Python, safe to call anywhere
install_packages()      -- Slicer-only, called only from explicit user action
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

TORCH_PACKAGES = ["torch", "torchvision"]
TORCH_INDEX    = "https://download.pytorch.org/whl/cpu"


def _is_importable(name: str) -> bool:
    import importlib.util
    import_name = {
        "scikit-image":                "skimage",
        "opencv-python-headless":      "cv2",
        "huggingface_hub":             "huggingface_hub",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
    }.get(name, name)
    return importlib.util.find_spec(import_name) is not None


def _numpy_major() -> int:
    """Return installed numpy major version, or 0 on failure."""
    try:
        import numpy as np
        return int(np.__version__.split(".")[0])
    except Exception:
        return 0


def get_missing_packages() -> dict:
    """
    Return {"torch": [...], "general": [...], "numpy_pin": ["numpy<2"] | []}.
    Pure Python — no slicer/qt import. Safe at any call site.
    """
    return {
        "torch":     [p for p in TORCH_PACKAGES   if not _is_importable(p)],
        "general":   [p for p in REQUIRED_PACKAGES if not _is_importable(p)],
        "numpy_pin": ["numpy<2"] if _numpy_major() >= 2 else [],
    }


def _pip_install(args_str: str) -> None:
    """Run pip install via subprocess, bypassing slicer.util.pip_install.

    This avoids the automatic Slicer pip-progress dialog that appears in
    Slicer 5.12+ when slicer.util.pip_install is called.
    Raises RuntimeError on non-zero exit.
    """
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + args_str.split(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])


def install_packages(missing: dict, pip_fn=None) -> None:
    """
    Install missing packages via subprocess pip, showing only our custom progress dialog.
    Only called from an explicit user action. Never called at setup time.
    missing: dict from get_missing_packages()
    pip_fn: injectable for testing (default: _pip_install)
    """
    import slicer
    if slicer.app.testingEnabled():
        return

    if pip_fn is None:
        pip_fn = _pip_install

    import qt
    import logging

    # Build step list for progress dialog
    steps = []
    if missing.get("torch"):
        steps.append(("torch+torchvision", "torch torchvision --index-url " + TORCH_INDEX))
    for pkg in missing.get("general", []):
        steps.append((pkg, pkg))
    # numpy_pin step added conditionally at the end

    total = len(steps)  # numpy pin added below if applicable

    progress = qt.QProgressDialog("Installing dependencies…", None, 0, total + 1)
    progress.setWindowTitle("ZebrafishAnalysis — Dependency Setup")
    progress.setMinimumWidth(400)
    progress.setWindowModality(qt.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    slicer.app.processEvents()

    errors = []
    torch_ok = False

    step = 0
    if missing.get("torch"):
        progress.setLabelText(f"Installing PyTorch (CPU)… ({step + 1}/{total + 1})")
        progress.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn("torch torchvision --index-url " + TORCH_INDEX)
            torch_ok = True
        except Exception as exc:
            logging.exception("Failed to install torch: %s", exc)
            errors.append(f"torch/torchvision: {exc}")
        step += 1

    for pkg in missing.get("general", []):
        progress.setLabelText(f"Installing {pkg}… ({step + 1}/{total + 1})")
        progress.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn(pkg)
        except Exception as exc:
            logging.exception("Failed to install %s: %s", pkg, exc)
            errors.append(f"{pkg}: {exc}")
        step += 1

    # Pin numpy<2 if torch was just installed successfully OR was already present
    already_has_torch = _is_importable("torch")
    if missing.get("numpy_pin") and (torch_ok or already_has_torch):
        progress.setLabelText(f"Pinning NumPy<2 for PyTorch compatibility… ({step + 1}/{total + 1})")
        progress.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn('numpy<2')
        except Exception as exc:
            logging.exception("Failed to pin numpy<2: %s", exc)
            errors.append(f"numpy<2: {exc}")
        step += 1

    progress.setValue(total + 1)
    progress.close()

    if errors:
        slicer.util.errorDisplay(
            "Some packages could not be installed:\n" + "\n".join(f"  • {e}" for e in errors)
        )
    else:
        slicer.util.showStatusMessage("ZebrafishAnalysis: dependencies installed — restart required.")
