"""
Logic layer for the ZebrafishEmbryoAnalyzer Slicer extension.

Wraps ZebrafishEmbryoAnalyzerCore functions and provides a clean public API:
  - analyse_images()   — batch segmentation + measurements
  - detect_scalebar()  — thin wrapper around core scalebar
  - calibrate_scale_from_points() — manual scale from two clicked bar ends

ZebrafishEmbryoAnalyzerCore and ZebrafishEmbryoAnalyzerLib import as packages because Slicer
puts the module directory on sys.path; no path manipulation here.
Export functions (export_excel, export_csv) live in export.py.
"""

import os
import shutil
import tempfile

_ML_PACKAGES = ("torch", "cv2", "segmentation_models_pytorch", "timm")

# Only reachable through an inconsistent request — the widget greys the Edema
# checkbox out for presets without an edema model (currently none). Raising
# beats skipping silently, which would report a successful run with the edema
# column quietly empty.
_EDEMA_UNAVAILABLE = (
    "Edema segmentation is not available for the {model_id!r} model. "
    "Choose a model that includes edema segmentation."
)


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
    import zebrafish_measure.seg as _seg_module
    # The library's segmentation_pipeline resolves _load_unet_model from the
    # zebrafish_measure.seg globals, so that is the module to patch.
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
                       revision="main", force_download=False, encoder_name="vgg16",
                       model_type="Unet", local_only=False, device=None):
    """Caching wrapper: first call loads from disk, subsequent calls return cached model."""
    cache_key = f"_unet_{model_path or filename}_{encoder_name}_{model_type}_{bool(local_only)}_{device}"
    if force_download or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _original_load_unet(
            model_path=model_path, repo_id=repo_id, filename=filename,
            label=label, revision=revision, force_download=force_download,
            encoder_name=encoder_name, model_type=model_type,
            local_only=local_only, device=device,
        )
    return _MODEL_CACHE[cache_key]


def preload_models(params: dict) -> None:
    """Load and cache all models needed for the given params.

    Called only after explicit Run Analysis.  Performs model weight
    deserialization synchronously in the current process.
    """
    _install_model_cache()
    from ZebrafishEmbryoAnalyzerCore.length import load_model
    from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError, ModelNotCachedError
    from ZebrafishEmbryoAnalyzerLib.model_manifest import (
        MODEL_SETS, get_cached_path, MODELS, verify_checksum,
    )

    model_id = params.get("model_id", "general")
    model_set = MODEL_SETS.get(model_id, MODEL_SETS["general"])
    # Validate the request before touching the filesystem, so an impossible ask
    # is reported as such rather than as a missing download.
    if params.get("edema", False) and "edema" not in model_set:
        raise AnalysisInputError(_EDEMA_UNAVAILABLE.format(model_id=model_id))

    if params.get("curvature", True) and "curvature" not in _MODEL_CACHE:
        curvature_entry = MODELS["curvature"]
        curvature_path = get_cached_path(curvature_entry)
        if not verify_checksum(curvature_path, curvature_entry["sha256"]):
            raise ModelNotCachedError(
                f"{curvature_entry['label']} missing or corrupted at {curvature_path}. "
                "Download models first."
            )
        _MODEL_CACHE["curvature"] = load_model(str(curvature_path))

    body_entry = model_set["body"]
    body_path = get_cached_path(body_entry)
    if not verify_checksum(body_path, body_entry["sha256"]):
        raise ModelNotCachedError(
            f"{body_entry['label']} missing or corrupted at {body_path}. "
            "Download models first."
        )
    _cached_load_unet(
        model_path=str(body_path),
        label="body model",
        encoder_name=body_entry["encoder"],
        model_type=body_entry.get("model_type", "Unet"),
        local_only=True,
        device="cpu",
    )

    if params.get("eyes", False):
        eye_entry = model_set["eye"]
        eye_path = get_cached_path(eye_entry)
        if not verify_checksum(eye_path, eye_entry["sha256"]):
            raise ModelNotCachedError(
                f"{eye_entry['label']} missing or corrupted at {eye_path}. "
                "Download models first."
            )
        _cached_load_unet(
            model_path=str(eye_path),
            label="eye model",
            encoder_name=eye_entry["encoder"],
            model_type=eye_entry.get("model_type", "Unet"),
            local_only=True,
            device="cpu",
        )

    if params.get("edema", False):
        edema_entry = model_set["edema"]
        edema_path = get_cached_path(edema_entry)
        if not verify_checksum(edema_path, edema_entry["sha256"]):
            raise ModelNotCachedError(
                f"{edema_entry['label']} missing or corrupted at {edema_path}. "
                "Download models first."
            )
        _cached_load_unet(
            model_path=str(edema_path),
            label="edema model",
            encoder_name=edema_entry["encoder"],
            model_type=edema_entry.get("model_type", "Unet"),
            local_only=True,
            device="cpu",
        )

    if params.get("swimbladder", False):
        swim_entry = model_set["swimbladder"]
        swim_path = get_cached_path(swim_entry)
        if not verify_checksum(swim_path, swim_entry["sha256"]):
            raise ModelNotCachedError(
                f"{swim_entry['label']} missing or corrupted at {swim_path}. "
                "Download models first."
            )
        _cached_load_unet(
            model_path=str(swim_path),
            label="swim bladder model",
            encoder_name=swim_entry["encoder"],
            model_type=swim_entry.get("model_type", "Unet"),
            local_only=True,
            device="cpu",
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
    "edema_mask",
    "swimbladder_mask",
    "path_points",
    "straight_line_points",
    "length",
    "curvature",
    "ratio",
    "eye_area",
    "eye_diameter",
    "edema_area",
    "swim_area",
    "swim_width",
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


_SCALE_MIN_UM_PER_PX = 0.001   # same range as the µm/px field / run_analysis
_SCALE_MAX_UM_PER_PX = 9999.0


def calibrate_scale_from_points(image_path: str, p1_rc, p2_rc, bar_um) -> dict:
    """
    Compute µm/px from two manually clicked scale-bar ends.

    p1_rc, p2_rc are (row, col) in original-image pixels (as delivered by the
    image viewer); bar_um is the physical bar length in µm.  Never raises:
    returns {success, scale_um_per_px, bar_length_px, message}.
    """
    import math

    def _fail(message, bar_px=None):
        return {"success": False, "scale_um_per_px": None,
                "bar_length_px": bar_px, "message": message}

    try:
        bar_um = float(bar_um)
    except (TypeError, ValueError):
        return _fail("Enter the physical bar length (µm) as a number.")
    if not math.isfinite(bar_um) or bar_um <= 0:
        return _fail("The physical bar length must be a positive number.")
    try:
        (r1, c1), (r2, c2) = p1_rc, p2_rc
        pt1 = (float(c1), float(r1))   # (row, col) -> (x, y)
        pt2 = (float(c2), float(r2))
    except (TypeError, ValueError):
        return _fail("Two valid points are required.")
    try:
        import cv2  # deferred: heavy compiled extension, only needed at call time
        from ZebrafishEmbryoAnalyzerCore.scalebar import calibrate_from_endpoints
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return _fail("Could not read image.")
        h, w = img_bgr.shape[:2]
        res = calibrate_from_endpoints(pt1, pt2, (h, w), label_um=bar_um)
    except Exception as exc:  # never raise into the GUI
        return _fail(f"Manual calibration failed: {exc}")
    if not res.get("bar_found", True):
        return _fail(res.get("message") or "Endpoints are too close together.")
    scale = res.get("scale_um_per_px")
    bar_px = res.get("bar_length_px")
    if not res.get("success") or scale is None:
        return _fail(res.get("message") or "Calibration failed.", bar_px)
    if not (_SCALE_MIN_UM_PER_PX <= scale <= _SCALE_MAX_UM_PER_PX):
        return _fail(
            f"Resulting scale {scale:.4g} µm/px is outside the allowed range "
            f"({_SCALE_MIN_UM_PER_PX} to {_SCALE_MAX_UM_PER_PX}). "
            "Check the clicked points and the bar length.", bar_px)
    return {"success": True, "scale_um_per_px": float(scale),
            "bar_length_px": float(bar_px), "message": res.get("message", "")}


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
          length, curvature, ratio, eyes, edema : bool
          hitl                           : bool  — use confidence threshold
          threshold                      : float 0–1
          um_per_px                      : float — physical scale (µm/pixel)
          model_id                       : str   — "fast" or "general"
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
        compute_tube_metrics,
    )
    from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError, ModelNotCachedError
    from ZebrafishEmbryoAnalyzerLib.model_manifest import (
        MODEL_SETS, MODEL_TARGET_SIZE, get_cached_path, MODELS, verify_checksum,
    )

    um_per_px = float(params.get("um_per_px", 22.99))
    include_eyes = params.get("eyes", False)
    include_edema = params.get("edema", False)
    include_swimbladder = params.get("swimbladder", False)

    model_id = params.get("model_id", "general")
    model_set = MODEL_SETS.get(model_id, MODEL_SETS["general"])
    target_size = MODEL_TARGET_SIZE.get(model_id, MODEL_TARGET_SIZE["general"])
    # Validate the request before touching the filesystem, so an impossible ask
    # is reported as such rather than as a missing download.
    if include_edema and "edema" not in model_set:
        raise AnalysisInputError(_EDEMA_UNAVAILABLE.format(model_id=model_id))

    # ---- validate required model files exist and are not corrupted before starting ----
    body_entry = model_set["body"]
    body_path = get_cached_path(body_entry)
    if not verify_checksum(body_path, body_entry["sha256"]):
        raise ModelNotCachedError(
            f"{body_entry['label']} missing or corrupted at {body_path}. Download models first."
        )
    if include_eyes:
        eye_entry = model_set["eye"]
        eye_path = get_cached_path(eye_entry)
        if not verify_checksum(eye_path, eye_entry["sha256"]):
            raise ModelNotCachedError(
                f"{eye_entry['label']} missing or corrupted at {eye_path}. Download models first."
            )
    if include_edema:
        edema_entry = model_set["edema"]
        edema_path = get_cached_path(edema_entry)
        if not verify_checksum(edema_path, edema_entry["sha256"]):
            raise ModelNotCachedError(
                f"{edema_entry['label']} missing or corrupted at {edema_path}. Download models first."
            )
    if include_swimbladder:
        swim_entry = model_set["swimbladder"]
        swim_path = get_cached_path(swim_entry)
        if not verify_checksum(swim_path, swim_entry["sha256"]):
            raise ModelNotCachedError(
                f"{swim_entry['label']} missing or corrupted at {swim_path}. Download models first."
            )
    if params.get("curvature", True):
        curv_entry = MODELS["curvature"]
        curv_manifest_path = get_cached_path(curv_entry)
        if not verify_checksum(curv_manifest_path, curv_entry["sha256"]):
            raise ModelNotCachedError(
                f"{curv_entry['label']} missing or corrupted at {curv_manifest_path}. "
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
        target_size=target_size,
        include_eyes=include_eyes,
        include_edema=include_edema,
        include_swimbladder=include_swimbladder,
        body_model_path=str(body_path),
        body_model_filename=body_entry["filename"],
        body_encoder_name=body_entry["encoder"],
        body_model_type=body_entry.get("model_type", "Unet"),
        local_only=True,
        device="cpu",
        swimbladder_biggest_only=True,
        swimbladder_boundary_offset_px=-0.73,
    )
    if include_eyes:
        _seg_kwargs["eye_model_path"] = str(eye_path)
        _seg_kwargs["eye_model_filename"] = eye_entry["filename"]
        _seg_kwargs["eye_encoder_name"] = eye_entry["encoder"]
        _seg_kwargs["eye_model_type"] = eye_entry.get("model_type", "Unet")
    if include_edema:
        _seg_kwargs["edema_model_path"] = str(edema_path)
        _seg_kwargs["edema_model_filename"] = edema_entry["filename"]
        _seg_kwargs["edema_encoder_name"] = edema_entry["encoder"]
        _seg_kwargs["edema_model_type"] = edema_entry.get("model_type", "Unet")
    if include_swimbladder:
        _seg_kwargs["swimbladder_model_path"] = str(swim_path)
        _seg_kwargs["swimbladder_model_filename"] = swim_entry["filename"]
        _seg_kwargs["swimbladder_encoder_name"] = swim_entry["encoder"]
        _seg_kwargs["swimbladder_model_type"] = swim_entry.get("model_type", "Unet")

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

            # seg_result is (originals, masks, growns) followed by whichever of
            # eyes/edema/swimbladder were requested, in that fixed order — mirrors
            # segmentation_pipeline's own conditional-append return (see seg.py).
            originals_bgr, masks, growns = seg_result[0], seg_result[1], seg_result[2]
            _extra = list(seg_result[3:])
            eyes_list = _extra.pop(0) if include_eyes else [None]
            edema_list = _extra.pop(0) if include_edema else [None]
            swimbladder_list = _extra.pop(0) if include_swimbladder else [None]

            orig_bgr = originals_bgr[0] if originals_bgr else None
            mask    = masks[0]      if masks      else None
            grown   = growns[0]     if growns     else None
            eye     = eyes_list[0]  if eyes_list  else None
            edema   = edema_list[0] if edema_list else None
            swim    = swimbladder_list[0] if swimbladder_list else None

            if orig_bgr is None:
                r["error"] = "Could not read image."
                results.append(r)
                if progress_callback:
                    progress_callback(_loop_i + 1, n)
                continue

            r["original"]   = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            r["mask"]             = mask
            r["grown"]            = grown
            r["eye_mask"]         = eye
            r["edema_mask"]       = edema
            r["swimbladder_mask"] = swim

            mask_bin  = (mask  > 0) if mask  is not None else None
            eye_bin   = (eye   > 0) if eye   is not None else None
            edema_bin = (edema > 0) if edema is not None else None

            h_orig, w_orig = orig_bgr.shape[:2]
            mask_h, mask_w = mask.shape[:2] if mask is not None else target_size
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
                        eye_bin, mask_fish=mask_bin, spacing=spacing,
                        area_shrink_frac=0.05,
                    )
                    r["eye_area"]     = float(info.get("eye_area",     0))
                    r["eye_diameter"] = float(info.get("eye_diameter", 0))
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Eye metrics error: {exc}"

            # ---- edema metrics ----
            if params.get("edema", False) and edema_bin is not None:
                try:
                    r["edema_area"] = float(np.count_nonzero(edema_bin) * spacing[0] * spacing[1])
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Edema metrics error: {exc}"

            # ---- swim bladder metrics ----
            if params.get("swimbladder", False) and swim is not None:
                try:
                    swim_metrics = compute_tube_metrics(swim, spacing=spacing, mask_fish=mask_bin)
                    r["swim_area"] = swim_metrics["area"]
                    r["swim_width"] = swim_metrics["width"]
                except Exception as exc:
                    if r["error"] is None:
                        r["error"] = f"Swim bladder metrics error: {exc}"

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
