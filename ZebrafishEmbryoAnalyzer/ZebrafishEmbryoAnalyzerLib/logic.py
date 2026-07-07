"""
Logic layer for the ZebrafishEmbryoAnalyzer Slicer extension.

Wraps ZebrafishEmbryoAnalyzerCore functions and provides a clean public API:
  - analyse_images()   — batch segmentation + measurements
  - detect_scalebar()  — thin wrapper around core scalebar

ZebrafishEmbryoAnalyzerCore and ZebrafishEmbryoAnalyzerLib import as packages because Slicer
puts the module directory on sys.path; no path manipulation here.
Export functions (export_excel, export_csv) live in export.py.
"""

import os
import shutil
import tempfile

_ML_PACKAGES = ("torch", "cv2", "segmentation_models_pytorch", "timm")


def dependency_status() -> dict:
    """Return availability of optional ML/vision dependencies.

    Performs lightweight importlib.util.find_spec checks — does NOT import the
    packages.  Suitable for calling at module load time without triggering heavy
    imports.

    Returns
    -------
    dict[str, bool]
        Keys are package names; values are True if the package is locatable on
        sys.path, False otherwise.
    """
    import importlib.util
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in _ML_PACKAGES}


_MODEL_CACHE: dict = {}
_original_load_unet = None  # set on first use by _install_model_cache()


def _install_model_cache():
    """Lazily import seg and install the caching monkey-patch (first call only)."""
    global _original_load_unet
    if _original_load_unet is not None:
        return
    import numpy as np  # noqa: F401 — must precede torch to enable numpy bridge  # noqa: F401
    import ZebrafishEmbryoAnalyzerCore.seg as _seg_module
    # On Slicer module reload, logic.py globals reset but seg._load_unet_model
    # still holds the wrapper from the old instance → would recurse infinitely.
    # Stash the true original on seg so it survives logic.py reloads.
    if hasattr(_seg_module, "_load_unet_model_original"):
        _original_load_unet = _seg_module._load_unet_model_original
    else:
        _original_load_unet = _seg_module._load_unet_model
        _seg_module._load_unet_model_original = _original_load_unet
    _seg_module._load_unet_model = _cached_load_unet


def _cached_load_unet(model_path=None, repo_id=None, filename=None, label="model",
                       revision="main", force_download=False, encoder_name="vgg16"):
    """Caching wrapper: first call loads from disk, subsequent calls return cached model."""
    cache_key = f"_unet_{model_path or filename}_{encoder_name}"
    if force_download or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _original_load_unet(
            model_path=model_path, repo_id=repo_id, filename=filename,
            label=label, revision=revision, force_download=force_download,
            encoder_name=encoder_name,
        )
    return _MODEL_CACHE[cache_key]


def preload_models(params: dict) -> None:
    """Load and cache all models needed for the given params.

    Called only after explicit Run Analysis.  Performs model weight
    deserialization synchronously in the current process.
    """
    _install_model_cache()
    from ZebrafishEmbryoAnalyzerCore.length import load_model
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_SETS, get_cached_path, MODELS

    model_id = params.get("model_id", "general")
    model_set = MODEL_SETS.get(model_id, MODEL_SETS["general"])

    if params.get("curvature", True) and "curvature" not in _MODEL_CACHE:
        curvature_entry = MODELS["curvature"]
        curvature_path = get_cached_path(curvature_entry)
        if not curvature_path.exists():
            raise ModelNotCachedError(
                f"{curvature_entry['label']} not found at {curvature_path}. "
                "Download models first."
            )
        _MODEL_CACHE["curvature"] = load_model(str(curvature_path))

    body_entry = model_set["body"]
    body_path = get_cached_path(body_entry)
    if not body_path.exists():
        raise ModelNotCachedError(
            f"{body_entry['label']} not found at {body_path}. "
            "Download models first."
        )
    _cached_load_unet(
        model_path=str(body_path),
        label="body model",
        encoder_name=body_entry["encoder"],
    )

    if params.get("eyes", False):
        eye_entry = model_set["eye"]
        eye_path = get_cached_path(eye_entry)
        if not eye_path.exists():
            raise ModelNotCachedError(
                f"{eye_entry['label']} not found at {eye_path}. "
                "Download models first."
            )
        _cached_load_unet(
            model_path=str(eye_path),
            label="eye model",
            encoder_name=eye_entry["encoder"],
        )

# ---------------------------------------------------------------------------
# Result dict schema — every key must be present, missing values use None
# ---------------------------------------------------------------------------
_RESULT_KEYS = (
    "filename",
    "image_path",
    "original",
    "mask",
    "grown",
    "eye_mask",
    "path_points",
    "straight_line_points",
    "length",
    "curvature",
    "ratio",
    "eye_area",
    "eye_diameter",
    "spacing",
    "error",
)


def _empty_result(image_path: str) -> dict:
    r = {k: None for k in _RESULT_KEYS}
    r["filename"] = os.path.basename(image_path)
    r["image_path"] = image_path
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scalebar(image_path: str, label_um: float | None = None) -> dict:
    """
    Detect scale bar in an image file.

    Returns the dict produced by core detect_scalebar, or a failure dict
    if the image cannot be read.
    """
    import cv2  # deferred: heavy compiled extension, only needed at call time
    from ZebrafishEmbryoAnalyzerCore.scalebar import detect_scalebar as _detect_scalebar
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"success": False, "bar_found": False,
                "message": "Could not read image."}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _detect_scalebar(img_rgb, label_um=label_um)


def analyse_images(image_paths: list, params: dict,
                   progress_callback=None) -> list:
    """
    Run segmentation + measurements on a list of image paths.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to input images.
    params : dict
        Keys:
          length, curvature, ratio, eyes : bool
          hitl                           : bool  — use confidence threshold
          threshold                      : float 0–1
          um_per_px                      : float — physical scale (µm/pixel)
          model_id                       : str   — "general" or "desy"
    progress_callback : callable(current, total) | None

    Returns
    -------
    list[dict]
        One result dict per image. Every dict contains all keys from the
        schema. On per-image errors the numeric fields are None and
        ``error`` holds the exception message.
    """
    import cv2  # deferred: heavy compiled extension, only needed at call time
    import numpy as np  # deferred: only needed inside analyse_images
    _install_model_cache()
    from ZebrafishEmbryoAnalyzerCore.seg import segmentation_pipeline
    from ZebrafishEmbryoAnalyzerCore.length import (
        load_model,
        tube_length_border2border,
        classification_curvature,
        compute_eye_metrics,
    )
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_SETS, get_cached_path, MODELS

    um_per_px = float(params.get("um_per_px", 22.99))
    include_eyes = params.get("eyes", False)

    model_id = params.get("model_id", "general")
    model_set = MODEL_SETS.get(model_id, MODEL_SETS["general"])

    # ---- validate required model files exist before starting ----
    body_entry = model_set["body"]
    body_path = get_cached_path(body_entry)
    if not body_path.exists():
        raise ModelNotCachedError(
            f"{body_entry['label']} not found at {body_path}. Download models first."
        )
    if include_eyes:
        eye_entry = model_set["eye"]
        eye_path = get_cached_path(eye_entry)
        if not eye_path.exists():
            raise ModelNotCachedError(
                f"{eye_entry['label']} not found at {eye_path}. Download models first."
            )
    if params.get("curvature", True):
        curv_entry = MODELS["curvature"]
        curv_manifest_path = get_cached_path(curv_entry)
        if not curv_manifest_path.exists():
            raise ModelNotCachedError(
                f"{curv_entry['label']} not found at {curv_manifest_path}. "
                "Download models first."
            )

    # ---- load curvature model once (cached across calls) ----
    if params.get("curvature", True):
        if "curvature" not in _MODEL_CACHE:
            _MODEL_CACHE["curvature"] = load_model(str(curv_manifest_path))
        curv_model = _MODEL_CACHE["curvature"]
    else:
        curv_model = None

    # ---- per-image segmentation + measurement ----
    # Call segmentation_pipeline once per image so progress_callback fires
    # after each one, keeping the UI responsive. The cached _load_unet_model
    # means model weights are only read from disk once across all calls.
    _seg_kwargs = dict(
        include_eyes=include_eyes,
        body_model_path=str(body_path),
        body_encoder_name=body_entry["encoder"],
    )
    if include_eyes:
        _seg_kwargs["eye_model_path"] = str(eye_path)

    n = len(image_paths)
    results = []

    for _loop_i, image_path in enumerate(sorted(image_paths)):
        r = _empty_result(image_path)

        try:
            # Segment this single image — model already cached, no disk reload
            with tempfile.TemporaryDirectory() as _tmp:
                # copy2 instead of symlink: os.symlink requires admin rights on
                # Windows; copy2 is portable and the temp dir is cleaned up
                # immediately after segmentation_pipeline returns.
                shutil.copy2(image_path, os.path.join(_tmp, os.path.basename(image_path)))
                seg_result = segmentation_pipeline(_tmp, **_seg_kwargs)

            if include_eyes and len(seg_result) == 4:
                originals_bgr, masks, growns, eyes_list = seg_result
            else:
                originals_bgr, masks, growns = seg_result[:3]
                eyes_list = [None]

            orig_bgr = originals_bgr[0] if originals_bgr else None
            mask    = masks[0]      if masks      else None
            grown   = growns[0]     if growns     else None
            eye     = eyes_list[0]  if eyes_list  else None

            if orig_bgr is None:
                r["error"] = "Could not read image."
                results.append(r)
                if progress_callback:
                    progress_callback(_loop_i + 1, n)
                continue

            r["original"]  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            r["mask"]      = mask
            r["grown"]     = grown
            r["eye_mask"]  = eye

            mask_bin = (mask > 0) if mask is not None else None
            eye_bin  = (eye  > 0) if eye  is not None else None

            h_orig, w_orig = orig_bgr.shape[:2]
            mask_h, mask_w = mask.shape[:2] if mask is not None else (256, 256)
            spacing = (
                um_per_px * h_orig / mask_h,
                um_per_px * w_orig / mask_w,
            )
            r["spacing"] = spacing

            # ---- length + ratio ----
            if params.get("length", True) and mask_bin is not None:
                try:
                    length, straight, path_pts, sl_pts = tube_length_border2border(
                        mask_bin,
                        spacing=spacing,
                        return_path=True,
                        return_straight_line=True,
                        mask_eye=eye_bin,
                        return_eye_info=False,
                    )
                    r["length"] = float(length)
                    r["path_points"] = path_pts
                    r["straight_line_points"] = sl_pts
                    if params.get("ratio", True) and straight and straight > 0:
                        r["ratio"] = float(length) / float(straight)
                except Exception as exc:
                    r["error"] = f"Length error: {exc}"

            # ---- curvature ----
            if params.get("curvature", True) and curv_model is not None:
                try:
                    use_thr = params.get("hitl", False)
                    thr = float(params.get("threshold", 0.85))
                    _, cls = classification_curvature(
                        orig_bgr, r["grown"], curv_model, use_thr, thr
                    )
                    r["curvature"] = int(cls.item())
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Curvature error: {exc}"

            # ---- eye metrics ----
            if params.get("eyes", False) and eye_bin is not None and mask_bin is not None:
                try:
                    info = compute_eye_metrics(
                        eye_bin, mask_fish=mask_bin, spacing=spacing
                    )
                    r["eye_area"]     = float(info.get("eye_area",     0))
                    r["eye_diameter"] = float(info.get("eye_diameter", 0))
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Eye metrics error: {exc}"

        except Exception as exc:
            import traceback
            r["error"] = f"Unhandled error: {exc}\n{traceback.format_exc()}"

        results.append(r)
        if progress_callback:
            progress_callback(_loop_i + 1, n)

    return results


# ---------------------------------------------------------------------------
# Manual correction
# ---------------------------------------------------------------------------

def apply_manual_correction(result, point1_orig, point2_orig, params=None):
    """
    Recompute length, ratio, and curvature from manually placed head/tail points.

    Parameters
    ----------
    result : dict
        Result dict (mutated in-place).  Must contain 'mask', 'original', 'spacing'.
    point1_orig, point2_orig : tuple
        (row, col) in original image coordinate space (as clicked on the display).
    params : dict | None
        Optional keys: 'hitl' (bool), 'threshold' (float).
        Used for curvature re-classification.  Defaults to hitl=False, threshold=0.85.

    Returns
    -------
    result : dict
        The same dict, updated in-place.
    """
    if params is None:
        params = {}

    spacing = result.get("spacing")
    if spacing is None:
        print("apply_manual_correction: spacing is None — skipping (fish had an error?)")
        return result

    mask = result.get("mask")
    original = result.get("original")
    if mask is None or original is None:
        print("apply_manual_correction: mask or original missing — skipping")
        return result

    import numpy as np  # deferred: only needed at call time
    from ZebrafishEmbryoAnalyzerCore.manual import compute_manual_length
    # Snapshot auto values on first correction only
    if "_auto_length" not in result:
        result["_auto_length"] = result.get("length")
        result["_auto_ratio"] = result.get("ratio")
        result["_auto_path_points"] = result.get("path_points")
        result["_auto_straight_line_points"] = result.get("straight_line_points")
        result["_auto_curvature"] = result.get("curvature")

    # Convert original-image coords → mask coords
    orig_h, orig_w = original.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    scale_y = mask_h / orig_h
    scale_x = mask_w / orig_w

    point1_mask = (
        int(np.clip(point1_orig[0] * scale_y, 0, mask_h - 1)),
        int(np.clip(point1_orig[1] * scale_x, 0, mask_w - 1)),
    )
    point2_mask = (
        int(np.clip(point2_orig[0] * scale_y, 0, mask_h - 1)),
        int(np.clip(point2_orig[1] * scale_x, 0, mask_w - 1)),
    )

    # Recompute length + path
    length, straight_length, path_pts, sl_pts = compute_manual_length(
        mask, point1_mask, point2_mask, spacing
    )
    result["length"] = float(length)
    result["ratio"] = float(length / straight_length) if straight_length > 0 else None
    result["path_points"] = path_pts
    result["straight_line_points"] = sl_pts

    result["manual_corrected"] = True
    return result


def revert_manual_correction(result):
    """
    Restore auto-computed values saved before the first manual correction.
    No-op if result['manual_corrected'] is not set.

    Returns
    -------
    result : dict
        The same dict, updated in-place.
    """
    if not result.get("manual_corrected"):
        return result

    result["length"] = result.pop("_auto_length", result.get("length"))
    result["ratio"] = result.pop("_auto_ratio", result.get("ratio"))
    result["path_points"] = result.pop("_auto_path_points", result.get("path_points"))
    result["straight_line_points"] = result.pop(
        "_auto_straight_line_points", result.get("straight_line_points")
    )
    result["curvature"] = result.pop("_auto_curvature", result.get("curvature"))
    result.pop("manual_corrected", None)
    return result
