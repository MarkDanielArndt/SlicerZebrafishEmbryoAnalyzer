"""Lazy re-export of ``zebrafish_measure.seg_helper`` (PyPI package ``zebrafish-measure``).

Importing this module does NOT import zebrafish_measure (nor torch/cv2/...);
a name is resolved from the library only when it is accessed.  Nothing is
cached on this module, so ``mock.patch`` on these names applies and restores
cleanly (the original is re-resolved from the library each time).
"""
import importlib

_NAMES = (
    "load_images_from_path",
    "segment_fish",
    "fill_holes",
    "grow_mask",
)


def __getattr__(name):
    if name in _NAMES:
        return getattr(importlib.import_module("zebrafish_measure.seg_helper"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_NAMES))
