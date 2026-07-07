"""
Characterization tests — lock down current behavior before structural refactoring.

These tests must not:
- require the full model download;
- require network access;
- require a graphical desktop;
- modify Slicer's Python environment.
"""

import os
import sys
import textwrap
import subprocess
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# Keys that widget, detail_tab, gallery_tab, export, and manual correction code
# access on every result dict.  Defined here independently of the production
# constant so a rename of _RESULT_KEYS does not silently remove the check.
_EXPECTED_RESULT_KEYS = frozenset({
    "filename", "image_path", "original", "mask", "grown", "eye_mask",
    "path_points", "straight_line_points", "length", "curvature", "ratio",
    "eye_area", "eye_diameter", "spacing", "error",
})


# ---------------------------------------------------------------------------
# Result schema — observable contract of analyse_images
# ---------------------------------------------------------------------------

def test_result_schema_all_keys_present(tmp_path, synthetic_fish_image, mock_model_paths):
    """Every result dict returned by analyse_images must contain all required keys."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipe, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"), \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_len, \
         patch("ZebrafishEmbryoAnalyzerCore.length.classification_curvature") as mock_curv:

        mock_pipe.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_mask.copy()],
        )
        mock_len.return_value = (1200.0, 1100.0, np.zeros((2, 2)), ((0, 0), (1, 1)))
        mock_cls = MagicMock()
        mock_cls.item.return_value = 2
        mock_curv.return_value = (None, mock_cls)

        results = analyse_images(
            [img_path],
            {"length": True, "curvature": True, "ratio": True,
             "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
        )

    assert len(results) == 1
    r = results[0]
    missing = _EXPECTED_RESULT_KEYS - r.keys()
    assert not missing, f"result dict missing required keys: {missing}"


def test_result_schema_preserved_on_per_image_error(tmp_path, synthetic_fish_image, mock_model_paths):
    """A per-image error must still produce a result dict with all required keys."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipe, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"), \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_len:

        mock_pipe.return_value = (
            [synthetic_fish_image[:, :, ::-1]], [dummy_mask], [dummy_mask.copy()]
        )
        mock_len.side_effect = RuntimeError("synthetic error")

        results = analyse_images(
            [img_path],
            {"length": True, "curvature": False, "ratio": False,
             "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
        )

    r = results[0]
    missing = _EXPECTED_RESULT_KEYS - r.keys()
    assert not missing, f"error result dict missing required keys: {missing}"
    assert r["error"] is not None


def test_result_filename_derived_from_path(tmp_path, synthetic_fish_image, mock_model_paths):
    """result['filename'] must be the basename of the input path."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images

    img_path = str(tmp_path / "sample_fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipe, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"), \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_len:

        mock_pipe.return_value = (
            [synthetic_fish_image[:, :, ::-1]], [dummy_mask], [dummy_mask.copy()]
        )
        mock_len.return_value = (1000.0, 950.0, np.zeros((2, 2)), ((0, 0), (1, 1)))

        results = analyse_images(
            [img_path],
            {"length": True, "curvature": False, "ratio": False,
             "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
        )

    assert results[0]["filename"] == "sample_fish.png"


# ---------------------------------------------------------------------------
# Progress callback contract
# ---------------------------------------------------------------------------

def test_progress_callback_called_once_per_image(tmp_path, synthetic_fish_image, mock_model_paths):
    """progress_callback(i, total) must be called exactly once per image."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images

    paths = []
    for i in range(3):
        p = str(tmp_path / f"fish_{i}.png")
        cv2.imwrite(p, synthetic_fish_image)
        paths.append(p)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    calls_received = []

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipe, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"), \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_len, \
         patch("ZebrafishEmbryoAnalyzerCore.length.classification_curvature") as mock_curv:

        mock_pipe.return_value = (
            [synthetic_fish_image[:, :, ::-1]], [dummy_mask], [dummy_mask.copy()]
        )
        mock_len.return_value = (1000.0, 900.0, np.zeros((2, 2)), ((0, 0), (1, 1)))
        mock_cls = MagicMock()
        mock_cls.item.return_value = 1
        mock_curv.return_value = (None, mock_cls)

        analyse_images(
            paths,
            {"length": True, "curvature": True, "ratio": True,
             "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
            progress_callback=lambda i, total: calls_received.append((i, total)),
        )

    assert len(calls_received) == 3
    totals = [t for _, t in calls_received]
    assert all(t == 3 for t in totals)
    indices = sorted(i for i, _ in calls_received)
    assert indices == [1, 2, 3]


# ---------------------------------------------------------------------------
# Model cache semantics
# ---------------------------------------------------------------------------

def test_model_cache_reused_across_preload_calls(tmp_path, mock_model_paths):
    """Calling preload_models twice with identical params must not reload model weights.

    The observable contract: the underlying load function (which reads from disk)
    is called exactly once across two preload_models calls with the same params.

    Setup: ensure the caching machinery is installed first (via _install_model_cache),
    then replace only logic._original_load_unet with a counting mock.  The finally
    block restores both touched module-level variables; _seg._load_unet_model is not
    touched so this test does not need to know about seg's internal state.
    """
    from ZebrafishEmbryoAnalyzerLib import logic

    body_path = str(tmp_path / "body.pth")
    (tmp_path / "body.pth").write_bytes(b"dummy")

    # Ensure the caching wrapper is installed on seg before we replace the inner fn.
    logic._install_model_cache()

    saved_original = logic._original_load_unet
    saved_cache = dict(logic._MODEL_CACHE)

    load_calls = [0]
    fake_model = MagicMock(name="fake_model")

    def counting_load(**kwargs):
        load_calls[0] += 1
        return fake_model

    try:
        logic._MODEL_CACHE.clear()
        logic._original_load_unet = counting_load

        params = {
            "curvature": False,
            "eyes": False,
            "body_model_filename": "body.pth",
            "body_encoder_name": "vgg19",
            "body_model_path": body_path,
        }
        with patch("ZebrafishEmbryoAnalyzerCore.length.load_model"):
            logic.preload_models(params)
            logic.preload_models(params)

        assert load_calls[0] == 1, (
            f"underlying load called {load_calls[0]} times; expected 1 (cache hit on second call)"
        )
    finally:
        # Restore exactly the two variables touched above; no seg-level state changed.
        logic._MODEL_CACHE.clear()
        logic._MODEL_CACHE.update(saved_cache)
        logic._original_load_unet = saved_original


# ---------------------------------------------------------------------------
# Import behavior without Slicer
# ---------------------------------------------------------------------------

def test_logic_lib_imports_without_slicer():
    """ZebrafishEmbryoAnalyzerLib.logic must be importable when 'slicer' is not available."""
    import ZebrafishEmbryoAnalyzerLib.logic as lg
    assert callable(lg.analyse_images)
    assert callable(lg.detect_scalebar)
    assert callable(lg.apply_manual_correction)
    assert callable(lg.revert_manual_correction)


def test_get_missing_packages_safe_outside_slicer():
    """get_missing_packages must return a valid dict without raising outside Slicer."""
    from ZebrafishEmbryoAnalyzerLib.dependency_installer import get_missing_packages
    result = get_missing_packages()
    assert isinstance(result, dict)
    assert "torch" in result
    assert "general" in result
    assert "numpy_pin" in result
    assert isinstance(result["torch"], list)
    assert isinstance(result["general"], list)
    assert isinstance(result["numpy_pin"], list)


# ---------------------------------------------------------------------------
# Qualified imports from packaged layout
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
)


def _run_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


def test_qualified_logic_import_from_package_layout():
    """From the packaged module directory, ZebrafishEmbryoAnalyzerLib.logic must import."""
    result = _run_subprocess("""
        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images, detect_scalebar
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_qualified_core_scalebar_manual_import_without_ml_deps():
    """scalebar and manual import without torch or segmentation_models_pytorch."""
    result = _run_subprocess("""
        import sys
        sys.modules["segmentation_models_pytorch"] = None
        from ZebrafishEmbryoAnalyzerCore import scalebar, manual
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_seg_helper_importable_without_torch_at_module_level():
    """After F1: seg must import without torch available at module level.

    Task F1 deferred torch to call time.  This test verifies the new behavior:
    seg imports successfully even when torch is blocked in sys.modules.
    """
    result = _run_subprocess("""
        import sys
        sys.modules["torch"] = None
        sys.modules["segmentation_models_pytorch"] = None
        sys.modules["huggingface_hub"] = None
        try:
            from ZebrafishEmbryoAnalyzerCore import seg
            print("IMPORTED_OK")
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"IMPORT_BLOCKED_UNEXPECTEDLY: {exc}")
    """)
    assert result.returncode == 0
    assert "IMPORTED_OK" in result.stdout, (
        "ZebrafishEmbryoAnalyzerCore.seg must be importable without torch at module level.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Export behavior
# ---------------------------------------------------------------------------

def test_export_csv_contains_expected_columns_and_values(tmp_path):
    """export_csv must produce a file with filename, length, curvature, ratio columns."""
    from ZebrafishEmbryoAnalyzerLib.export import export_csv

    results = [
        {
            "filename": "a.png", "length": 1200.0, "curvature": 2, "ratio": 1.05,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
        {
            "filename": "b.png", "length": None, "curvature": None, "ratio": None,
            "eye_area": None, "eye_diameter": None, "error": "unreadable",
        },
    ]
    out = tmp_path / "out.csv"
    export_csv(results, str(out))

    content = out.read_text()
    assert "a.png" in content
    assert "b.png" in content
    assert "1200" in content
    # Header row must include key column names
    first_line = content.splitlines()[0]
    assert "Filename" in first_line
    assert "Length" in first_line


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

def test_analyse_images_error_field_contains_message(tmp_path, synthetic_fish_image, mock_model_paths):
    """When segmentation_pipeline raises, the error field must contain the message."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipe, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"):

        mock_pipe.side_effect = RuntimeError("GPU out of memory")

        results = analyse_images(
            [img_path],
            {"length": True, "curvature": True, "ratio": True,
             "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
        )

    assert len(results) == 1
    assert results[0]["error"] is not None
    assert "GPU out of memory" in results[0]["error"]
