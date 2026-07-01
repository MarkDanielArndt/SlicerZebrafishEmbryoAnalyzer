"""
Dependency status check and install helpers for ZebrafishEmbryoAnalyzer.

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


def _pip_install(args_str: str, cancel_check=None) -> None:
    """Run pip install via subprocess, keeping the Qt event loop alive.

    Uses Popen + polling so the progress dialog stays responsive during long
    installs (Windows hides frozen dialogs; Linux shows 'not responding').
    Outside Slicer (unit tests), falls back to a blocking wait.
    Raises RuntimeError on non-zero exit or when cancel_check() returns True.
    """
    import subprocess
    import sys
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install"] + args_str.split(),
        stdout=subprocess.DEVNULL,  # discard pip progress output; prevents pipe-buffer deadlock
        stderr=subprocess.PIPE,
    )
    try:
        import slicer as _slicer
        import time
        while proc.poll() is None:
            _slicer.app.processEvents()
            if cancel_check and cancel_check():
                proc.terminate()
                raise RuntimeError("Installation cancelled by user.")
            time.sleep(0.05)
    except ImportError:
        proc.wait()  # outside Slicer (unit tests) — blocking wait is fine
    out, err = proc.communicate()
    if proc.returncode != 0:
        detail = (err or b"").decode(errors="replace")
        raise RuntimeError(detail[-2000:])


def install_packages(missing: dict, pip_fn=None) -> bool:
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

    prog_dlg = qt.QDialog(slicer.util.mainWindow())
    prog_dlg.setWindowTitle("ZebrafishEmbryoAnalyzer — Dependency Setup")
    prog_dlg.setWindowFlags(
        prog_dlg.windowFlags()
        & ~qt.Qt.WindowContextHelpButtonHint
        & ~qt.Qt.WindowCloseButtonHint
    )
    prog_dlg.setMinimumWidth(400)
    _prog_layout = qt.QVBoxLayout(prog_dlg)
    prog_label = qt.QLabel("Installing dependencies…")
    _prog_layout.addWidget(prog_label)
    prog_bar = qt.QProgressBar()
    prog_bar.setRange(0, total + 1)
    prog_bar.setValue(0)
    _prog_layout.addWidget(prog_bar)

    cancelled = [False]

    def _on_cancel():
        cancelled[0] = True

    _btn_box = qt.QDialogButtonBox(prog_dlg)
    _btn_box.addButton("Cancel", qt.QDialogButtonBox.RejectRole)
    _btn_box.connect("rejected()", _on_cancel)
    _prog_layout.addWidget(_btn_box)

    prog_dlg.show()
    slicer.app.processEvents()

    errors = []
    torch_ok = False

    step = 0
    if missing.get("torch"):
        prog_label.setText(f"Installing PyTorch (CPU)… ({step + 1}/{total + 1})")
        prog_bar.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn("torch torchvision --index-url " + TORCH_INDEX,
                   cancel_check=lambda: cancelled[0])
            torch_ok = True
        except Exception as exc:
            logging.exception("Failed to install torch: %s", exc)
            errors.append(f"torch/torchvision: {exc}")
        step += 1
        slicer.app.processEvents()
        if cancelled[0]:
            prog_dlg.close()
            return False

    for pkg in missing.get("general", []):
        prog_label.setText(f"Installing {pkg}… ({step + 1}/{total + 1})")
        prog_bar.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn(pkg, cancel_check=lambda: cancelled[0])
        except Exception as exc:
            logging.exception("Failed to install %s: %s", pkg, exc)
            errors.append(f"{pkg}: {exc}")
        step += 1
        slicer.app.processEvents()
        if cancelled[0]:
            break

    # Pin numpy<2 if torch was just installed successfully OR was already present
    if cancelled[0]:
        prog_dlg.close()
        return False
    already_has_torch = _is_importable("torch")
    if missing.get("numpy_pin") and (torch_ok or already_has_torch):
        prog_label.setText(f"Pinning NumPy<2 for PyTorch compatibility… ({step + 1}/{total + 1})")
        prog_bar.setValue(step)
        slicer.app.processEvents()
        try:
            pip_fn('numpy<2', cancel_check=lambda: cancelled[0])
        except Exception as exc:
            logging.exception("Failed to pin numpy<2: %s", exc)
            errors.append(f"numpy<2: {exc}")
        step += 1

    prog_bar.setValue(total + 1)
    prog_dlg.close()

    if errors:
        slicer.util.errorDisplay(
            "Some packages could not be installed:\n" + "\n".join(f"  • {e}" for e in errors)
        )
    else:
        slicer.util.showStatusMessage("ZebrafishEmbryoAnalyzer: dependencies installed — restart required.")

    return True
