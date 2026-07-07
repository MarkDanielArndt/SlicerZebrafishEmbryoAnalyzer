"""
Model manifest for ZebrafishEmbryoAnalyzer.

Pure Python — no slicer, qt, vtk, ctk, or torch imports.
Describes all model files, their provenance, and cache locations.

sha256 values are lowercase hex SHA-256 digests of the file at the pinned revision.
Revisions are immutable commit SHAs, not floating branch names.
size_bytes values are approximate estimates used only for pre-download UI;
real sizes come from Content-Length headers during download.
Licenses are "LICENSE_PENDING" until confirmed.
"""

import hashlib
import os
from pathlib import Path


def _default_cache_dir() -> Path:
    """Return a platform-appropriate cache directory for model files.

    Prefers ``platformdirs.user_cache_dir`` (cross-platform).  Falls back to
    ``~/.cache/zebrafish_models`` if platformdirs is not installed so that
    existing deployments on macOS/Linux keep working without an additional
    dependency.
    """
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("zebrafish_models"))
    except Exception:
        # Catch ImportError (platformdirs absent) and runtime errors such as
        # PermissionError / OSError on Windows with a restricted LOCALAPPDATA.
        return Path.home() / ".cache" / "zebrafish_models"


_CACHE_DIR = _default_cache_dir()

# ---------------------------------------------------------------------------
# All known model entries
# ---------------------------------------------------------------------------

MODELS: dict = {
    "general_body": {
        "id": "general_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_3400_vgg19.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "Body segmentation model",
        "encoder": "vgg19",
        "sha256": "624e9ef0ab447aee7b95a058596c048f033a8255bc850f3a238b5606ea71ae65",
        "size_bytes": 116_289_435,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_eye": {
        "id": "general_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_3400.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "Eye segmentation model",
        "encoder": "vgg16",
        "sha256": "026b799fef133ddc44237d9f70f52694b0a02708d84a20b1f5e718a414250a2e",
        "size_bytes": 95_048_239,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # Not yet wired into any MODEL_SET; kept for future edema analysis support.
    "general_edema": {
        "id": "general_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_3400_focal.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "Edema segmentation model",
        "encoder": "vgg19",
        "sha256": "3622392fc8a65d9de1f49554770422cf3661deee8381a4fbbd62c48d01c6dfaf",
        "size_bytes": 116_290_283,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "curvature": {
        "id": "curvature",
        "repo_id": "markdanielarndt/Classification",
        "filename": "best_model_class.pth",
        "revision": "926bea8cec2898e6eb313f8748318f6053876ed8",
        "label": "Curvature classification model",
        "encoder": None,
        "sha256": "7b9c029ed1b8887fca2fe42197d010422b8a822e3aa86da6ffcdbdb530ebdc6c",
        "size_bytes": 352_517_483,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_body": {
        "id": "desy_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_finetuned.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "DESY body segmentation model",
        "encoder": "vgg19",
        "sha256": "aff8eedbfcf682bd0fc72fb4dcf26f9fa3d3a0e4a5304cbc070fcc50e08476fc",
        "size_bytes": 116_290_059,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "desy_eye": {
        "id": "desy_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_finetuned.pth",
        "revision": "673bc5d60e786a8413ecefbcc1701e1ec6ed6ae1",
        "label": "DESY eye segmentation model",
        "encoder": "vgg16",
        "sha256": "fccd2da00d2ac7fbbde2dc18da51641f9e10eb905ed0a8a8fa498a8fe91c2690",
        "size_bytes": 95_048_833,
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

    Raises ValueError for missing, malformed, or placeholder hash values
    so that configuration errors are caught early rather than silently skipped.

    Returns True when the file hash matches sha256.
    Raises ValueError for placeholder ("PENDING") or empty sha256.
    Returns False when the file hash does not match or the file cannot be read.
    """
    if not sha256 or sha256 == "PENDING":
        raise ValueError(
            f"Model at {path!r} has a placeholder or missing SHA-256 checksum. "
            "Update model_manifest.py with the real checksum before using this model."
        )
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != sha256:
            return False
        return True
    except OSError:
        return False


def checksum_mismatch_error(entry: dict, path, actual_sha256: str) -> str:
    """Return a human-readable checksum mismatch error string."""
    return (
        f"Checksum mismatch for model {entry['id']!r} at {path!r}.\n"
        f"  Expected: {entry['sha256']}\n"
        f"  Actual:   {actual_sha256}\n"
        "The file may be corrupted or tampered. Delete it and re-download."
    )


def collect_all_model_entries() -> dict:
    """Return all unique model entries across all MODEL_SETS, deduplicated by id.

    Returns
    -------
    dict
        Mapping of entry id -> model_entry for every entry that appears in any
        MODEL_SET.  Entries shared across sets (e.g. curvature) appear once.
    """
    result = {}
    for variant in MODEL_SETS.values():
        for entry in variant.values():
            result.setdefault(entry["id"], entry)
    return result


def get_missing_models(model_set_dict: dict) -> list:
    """
    Return model entries whose cached path does not exist or is empty.

    Parameters
    ----------
    model_set_dict : dict
        Mapping of str -> model_entry, e.g. ``MODEL_SETS["general"]``.

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
