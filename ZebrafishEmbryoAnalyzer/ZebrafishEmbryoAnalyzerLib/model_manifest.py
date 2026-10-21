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
    # Mirrors two of the webapp's presets (SEG_MODEL_OPTIONS in its app.py):
    # "Fast & Easy" (256px) and "Complex & Slower" (512px — our "general").
    # The webapp's third preset (1024px) is webapp-only and not offered here.
    # Every segmentation role of both presets uses the webapp's Segformer
    # (mit_b3) models, resolution-matched per preset.
    "fast_body": {
        "id": "fast_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_256_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Body segmentation model (256 px, Fast & Easy)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "e4f04e97803a665839c75c70bdc75a3a89dbee28e72da1b640daf4ed2e2a2677",
        "size_bytes": 178_629_809,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "fast_eye": {
        "id": "fast_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_256_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Eye segmentation model (256 px, Fast & Easy)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "5bf80e5bb92912200614cb73d9ce3ff2c811d2441c8660c4336f6f8713a935e7",
        "size_bytes": 178_629_215,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "fast_edema": {
        "id": "fast_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_256_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Edema segmentation model (256 px, Fast & Easy)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "58ed666bb6540590ccade64027ee25fbde23bc950b6e5803a7bb8eccdecf043c",
        "size_bytes": 178_630_403,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "fast_swimbladder": {
        "id": "fast_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_swimmbladder_256_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Swim bladder segmentation model (256 px, Fast & Easy)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "8e62d653da75a996eb3802465f29ba3f2d7f5255c5411954583d143f91f184dd",
        "size_bytes": 178_634_561,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_body": {
        "id": "general_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_512_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Body segmentation model (512 px, Complex & Slower)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "54651c627195bb7291a65fe0808992565ff74b222925efc4c810ed637074e8fc",
        "size_bytes": 178_629_809,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_eye": {
        "id": "general_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_512_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Eye segmentation model (512 px, Complex & Slower)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "646fd95ac5dc6ecad6d4320c4349648e796b75e50b113a8f4efd023c108592e0",
        "size_bytes": 178_629_215,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_edema": {
        "id": "general_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_512_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Edema segmentation model (512 px, Complex & Slower)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "8f8f00305c2367d54dd114d0a316bdd4ad3251dae18f948cde648dbb48267ac0",
        "size_bytes": 178_630_403,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_swimbladder": {
        "id": "general_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_swimmbladder_512_segformer_mit_b3.pth",
        "revision": "993a5d31cb3dde4ac8af72a5cbee389e1c25f239",
        "label": "Swim bladder segmentation model (512 px, Complex & Slower)",
        "encoder": "mit_b3",
        "model_type": "Segformer",
        "sha256": "dd785146ec5125a4bb1a038b4ef5e8a25c99076b1c4fb007d68c85b89ea1942a",
        "size_bytes": 178_634_561,
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
}

# ---------------------------------------------------------------------------
# Model sets per variant: variant_id -> {role -> model_entry}
# ---------------------------------------------------------------------------

MODEL_SETS: dict = {
    # Both presets offer every role, including "edema" (resolution-matched
    # 256px / 512px models). widget.py's edema checkbox logic stays generic.
    "fast": {
        "body": MODELS["fast_body"],
        "eye": MODELS["fast_eye"],
        "curvature": MODELS["curvature"],
        "edema": MODELS["fast_edema"],
        "swimbladder": MODELS["fast_swimbladder"],
    },
    "general": {
        "body": MODELS["general_body"],
        "eye": MODELS["general_eye"],
        "curvature": MODELS["curvature"],
        "edema": MODELS["general_edema"],
        "swimbladder": MODELS["general_swimbladder"],
    },
}

# Per-preset segmentation_pipeline() target_size, matching the webapp's own
# per-preset resolution (see SEG_MODEL_OPTIONS in the webapp's app.py). Kept
# separate from MODEL_SETS so every value there stays a model-entry dict —
# get_missing_models()/verify_checksum() iterate MODEL_SETS[...].values()
# expecting exactly that shape.
MODEL_TARGET_SIZE: dict = {
    "fast": (256, 256),
    "general": (512, 512),
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
    Return model entries whose cached file is missing or fails checksum verification.

    A truncated or corrupted download (present on disk, non-zero size, wrong
    content — e.g. an interrupted transfer) must be treated the same as a
    genuinely missing file. Otherwise it looks "cached" until deserialization
    fails deep inside analysis with a cryptic pickle error instead of the normal
    download-prompt flow.

    Parameters
    ----------
    model_set_dict : dict
        Mapping of str -> model_entry, e.g. ``MODEL_SETS["general"]``.

    Returns
    -------
    list[dict]
        Subset of model_set_dict.values() that are not yet correctly cached.
    """
    return [
        entry for entry in model_set_dict.values()
        if not verify_checksum(get_cached_path(entry), entry["sha256"])
    ]
