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
    # The webapp's third preset is webapp-only and not offered here.
    # In the webapp a `None` filename in that table means "use the pipeline
    # default", not "unavailable" — those defaults are the 256px files, which is
    # why Fast & Easy resolves to fast_eye/default_edema/fast_swimbladder below.
    "fast_body": {
        "id": "fast_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_3400_vgg19.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Body segmentation model (256 px, Fast & Easy)",
        "encoder": "vgg19",
        "sha256": "624e9ef0ab447aee7b95a058596c048f033a8255bc850f3a238b5606ea71ae65",
        "size_bytes": 116_289_435,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "fast_eye": {
        "id": "fast_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_3400.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Eye segmentation model (256 px, Fast & Easy)",
        "encoder": "vgg16",
        "sha256": "026b799fef133ddc44237d9f70f52694b0a02708d84a20b1f5e718a414250a2e",
        "size_bytes": 95_048_239,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # Unet + vgg16 here, unlike the 512px swim bladder models (FPN + vgg19) —
    # this is the webapp's own pipeline default, not an oversight.
    "fast_swimbladder": {
        "id": "fast_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_swimmbladder_256_09072026.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Swim bladder segmentation model (256 px, Fast & Easy)",
        "encoder": "vgg16",
        "model_type": "Unet",
        "sha256": "656dcefd46f5d1e53605f9d5a68a23a0fd6203b3daa16e61ac11ffbdeb5cd27b",
        "size_bytes": 95_049_151,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_body": {
        "id": "general_body",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_body_512.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Body segmentation model (512 px, Complex & Slower)",
        "encoder": "vgg19",
        "sha256": "030fd29623467a12eb5d913460fd778d8773345f98f152bbaeae1ebbbbb9ecf2",
        "size_bytes": 116_289_323,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    "general_eye": {
        "id": "general_eye",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_eye_512.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Eye segmentation model (512 px, Complex & Slower)",
        "encoder": "vgg16",
        "sha256": "9493f3ae60ecddbe1f16717b9c28b2a43491adc8b6d8d066031ae83d3b5cd383",
        "size_bytes": 95_048_133,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # The only edema model, and it is 256px-era: the repo has no 512px
    # general counterpart. The webapp falls back to it for its 512px preset too
    # (no edema filename named there -> pipeline default), feeding a 256px-trained
    # net 512px input. We deliberately do not follow that one step — see the
    # "edema" comment in MODEL_SETS below.
    "default_edema": {
        "id": "default_edema",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_edema_3400_focal.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Edema segmentation model",
        "encoder": "vgg19",
        "sha256": "3622392fc8a65d9de1f49554770422cf3661deee8381a4fbbd62c48d01c6dfaf",
        "size_bytes": 116_290_283,
        "license": "LICENSE_PENDING",
        "preprocessing_compat": "v1",
    },
    # FPN (segmentation_models_pytorch), not Unet like the other segmentation
    # roles — see model_type, consumed by seg.py's _load_unet_model.
    "general_swimbladder": {
        "id": "general_swimbladder",
        "repo_id": "markdanielarndt/Zebrafish_Segmentation",
        "filename": "best_model_swimmbladder_512_09072026.pth",
        "revision": "237d21d6d7538fc5b661bf43b70f378f945991ee",
        "label": "Swim bladder segmentation model (512 px, Complex & Slower)",
        "encoder": "vgg19",
        "model_type": "FPN",
        "sha256": "d11e41c7504bbd388f29b53d2a31a731190e4b1b26f036326f4a3c104334d5ab",
        "size_bytes": 88_459_594,
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
    # A preset offers "edema" only where a resolution-matched model exists:
    # fast pairs the 256px default with 256px input. "general" runs at 512px but no 512px general edema model exists, so
    # the role is omitted rather than silently running a 256px-trained net at
    # 512px the way the webapp does. widget.py greys the checkbox out there.
    "fast": {
        "body": MODELS["fast_body"],
        "eye": MODELS["fast_eye"],
        "curvature": MODELS["curvature"],
        "edema": MODELS["default_edema"],
        "swimbladder": MODELS["fast_swimbladder"],
    },
    "general": {
        "body": MODELS["general_body"],
        "eye": MODELS["general_eye"],
        "curvature": MODELS["curvature"],
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
