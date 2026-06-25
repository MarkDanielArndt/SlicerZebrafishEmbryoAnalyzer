"""
Model manifest for ZebrafishAnalysis.

Pure Python — no slicer, qt, vtk, ctk, or torch imports.
Describes all model files, their provenance, and cache locations.

SHA-256 values are "PENDING" until final model files are verified.
size_bytes values are approximate estimates used only for pre-download UI;
real sizes come from Content-Length headers during download.
Licenses are "LICENSE_PENDING" until confirmed.
"""

import hashlib
import os
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "zebrafish_models"

# ---------------------------------------------------------------------------
# All known model entries
# ---------------------------------------------------------------------------

MODELS: dict = {
    "general_body": {
        "id": "general_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_3400_vgg19.pth",
        "revision": "main",
        "label": "Body segmentation model",
        "encoder": "vgg19",
        "sha256": "PENDING",
        "size_bytes": 555_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_eye": {
        "id": "general_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_3400.pth",
        "revision": "main",
        "label": "Eye segmentation model",
        "encoder": "vgg16",
        "sha256": "PENDING",
        "size_bytes": 555_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_edema": {
        "id": "general_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_3400_focal.pth",
        "revision": "main",
        "label": "Edema segmentation model",
        "encoder": "vgg19",
        "sha256": "PENDING",
        "size_bytes": 555_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "curvature": {
        "id": "curvature",
        "repo_id": "markdanielarndt/Classification",
        "filename": "best_model_class.pth",
        "revision": "main",
        "label": "Curvature classification model",
        "encoder": None,
        "sha256": "PENDING",
        "size_bytes": 355_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_body": {
        "id": "desy_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_finetuned.pth",
        "revision": "main",
        "label": "DESY body segmentation model",
        "encoder": "vgg19",
        "sha256": "PENDING",
        "size_bytes": 555_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_eye": {
        "id": "desy_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_finetuned.pth",
        "revision": "main",
        "label": "DESY eye segmentation model",
        "encoder": "vgg16",
        "sha256": "PENDING",
        "size_bytes": 555_000_000,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
}

# ---------------------------------------------------------------------------
# Model sets per variant: variant_id -> {role -> model_entry}
# ---------------------------------------------------------------------------

MODEL_SETS: dict = {
    "general": {
        "body": MODELS["general_body"],
        "eye": MODELS["general_eye"],
        "curvature": MODELS["curvature"],
    },
    "desy": {
        "body": MODELS["desy_body"],
        "eye": MODELS["desy_eye"],
        "curvature": MODELS["curvature"],
    },
}


def get_cached_path(entry: dict) -> Path:
    """Return the local cache Path for a model entry."""
    return _CACHE_DIR / entry["filename"]


def verify_checksum(path, sha256: str) -> bool:
    """
    Verify SHA-256 checksum of a file.

    Returns True when sha256 == "PENDING" (skip check — hash not yet known).
    Returns True when the file hash matches sha256.
    Returns False when the file hash does not match or the file cannot be read.
    """
    if sha256 == "PENDING":
        return True
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == sha256
    except OSError:
        return False


def get_missing_models(model_set_dict: dict) -> list:
    """
    Return model entries whose cached path does not exist or is empty.

    Parameters
    ----------
    model_set_dict : dict
        Mapping of role -> model_entry, e.g. ``MODEL_SETS["general"]``.

    Returns
    -------
    list[dict]
        Subset of model_set_dict.values() that are not yet cached.
    """
    missing = []
    for entry in model_set_dict.values():
        p = get_cached_path(entry)
        try:
            if not p.exists() or p.stat().st_size == 0:
                missing.append(entry)
        except OSError:
            missing.append(entry)
    return missing
