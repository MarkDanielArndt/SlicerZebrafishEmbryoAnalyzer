"""Lazy re-export of ``zebrafish_measure.length`` (PyPI package ``zebrafish-measure``).

Importing this module does NOT import zebrafish_measure (nor torch/cv2/...);
a name is resolved from the library only when it is accessed.  Nothing is
cached on this module, so ``mock.patch`` on these names applies and restores
cleanly (the original is re-resolved from the library each time).
"""
import importlib

_NAMES = (
    "compute_eye_metrics",
    "compute_tube_metrics",
    "compute_eye_diameters",
    "tube_length_border2border",
    "normalize_images",
    "apply_mask",
    "select_torch_device",
    "classification_curvature",
    "load_model",
    "plot_edges_with_curvature",
    "compute_curvature_profile",
    "compute_curvature",
    "preprocess_masked_image",
)


def __getattr__(name):
    if name in _NAMES:
        return getattr(importlib.import_module("zebrafish_measure.length"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_NAMES))
