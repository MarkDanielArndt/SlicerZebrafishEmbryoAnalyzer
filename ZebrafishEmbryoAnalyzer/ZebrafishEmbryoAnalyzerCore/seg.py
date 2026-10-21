"""Lazy re-export of ``zebrafish_measure.seg`` (PyPI package ``zebrafish-measure``).

Importing this module does NOT import zebrafish_measure (nor torch/cv2/...);
a name is resolved from the library only when it is accessed.  Nothing is
cached on this module, so ``mock.patch`` on these names applies and restores
cleanly (the original is re-resolved from the library each time).
"""
import importlib

_NAMES = (
    "_UNET_CACHE",
    "_load_unet_model",
    "segmentation_pipeline",
)


def __getattr__(name):
    if name in _NAMES:
        return getattr(importlib.import_module("zebrafish_measure.seg"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_NAMES))
