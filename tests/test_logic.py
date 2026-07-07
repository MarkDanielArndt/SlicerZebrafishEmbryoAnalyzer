import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_seg_result(synthetic_fish_image, synthetic_fish_mask):
    bgr = synthetic_fish_image[:, :, ::-1].copy()
    return [{
        "filename":             "fish.png",
        "image_path":           "/tmp/fish.png",
        "original":             synthetic_fish_image,
        "mask":                 synthetic_fish_mask,
        "grown":                synthetic_fish_mask.copy(),
        "eye_mask":             None,
        "path_points":          np.array([[64, 128], [128, 128], [192, 128]]),
        "straight_line_points": ((64, 128), (192, 128)),
        "length":               1200.0,
        "curvature":            2,
        "ratio":                1.05,
        "eye_area":             None,
        "eye_diameter":         None,
    }]


def test_analyse_images_returns_one_result_per_image(tmp_path, synthetic_fish_image, mock_model_paths):
    import cv2
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_grown = dummy_mask.copy()

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as mock_load, \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_length, \
         patch("ZebrafishEmbryoAnalyzerCore.length.classification_curvature") as mock_curv:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_grown],
        )
        mock_load.return_value = MagicMock()
        mock_length.return_value = (1200.0, 1100.0,
                                    np.array([[64, 128], [192, 128]]),
                                    ((64, 128), (192, 128)))
        mock_curv.return_value = (MagicMock(), MagicMock(item=lambda: 2))

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        results = analyse_images(
            [img_path],
            {"length": True, "curvature": True, "ratio": True,
             "eyes": False, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99},
        )

    assert len(results) == 1
    assert results[0]["filename"] == "fish.png"
    assert results[0]["length"] == pytest.approx(1200.0)


def test_analyse_images_error_per_image_does_not_crash(tmp_path, synthetic_fish_image, mock_model_paths):
    import cv2
    p1 = str(tmp_path / "good.png")
    cv2.imwrite(p1, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model"), \
         patch("ZebrafishEmbryoAnalyzerCore.length.tube_length_border2border") as mock_length:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_mask.copy()],
        )
        mock_length.side_effect = RuntimeError("synthetic error")

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        results = analyse_images(
            [p1],
            {"length": True, "curvature": False, "ratio": False,
             "eyes": False, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99},
        )

    assert len(results) == 1
    assert results[0]["length"] is None


def test_models_to_download_all_cached(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.model_manifest import get_missing_models, MODEL_SETS
    model_set = MODEL_SETS["general"]
    for entry in model_set.values():
        f = tmp_path / entry["filename"]
        f.write_bytes(b"\x00")
    with patch(
        "ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
        side_effect=lambda entry: tmp_path / entry["filename"],
    ):
        result = get_missing_models(model_set)
    assert result == []


def test_models_to_download_missing(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.model_manifest import get_missing_models, MODEL_SETS
    model_set = MODEL_SETS["general"]
    with patch(
        "ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
        side_effect=lambda entry: tmp_path / entry["filename"],
    ):
        result = get_missing_models(model_set)
    assert len(result) == len(model_set)


# ---------------------------------------------------------------------------
# M1 — ModelNotCachedError raised when model file absent
# ---------------------------------------------------------------------------

def test_analyse_images_raises_model_not_cached_when_body_missing(tmp_path, synthetic_fish_image):
    """analyse_images must raise ModelNotCachedError when body model file is absent."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError
    from pathlib import Path

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    # get_cached_path returns a non-existent path — no files created
    def fake_get_cached_path(entry):
        return tmp_path / entry["filename"]

    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
               side_effect=fake_get_cached_path):
        with pytest.raises(ModelNotCachedError):
            analyse_images(
                [img_path],
                {"length": True, "curvature": False, "ratio": False,
                 "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
            )


def test_preload_models_raises_model_not_cached_when_body_missing(tmp_path):
    """preload_models must raise ModelNotCachedError when body model file is absent."""
    from ZebrafishEmbryoAnalyzerLib import logic
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError

    def fake_get_cached_path(entry):
        return tmp_path / entry["filename"]

    logic._install_model_cache()
    saved_cache = dict(logic._MODEL_CACHE)
    logic._MODEL_CACHE.clear()
    try:
        with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
                   side_effect=fake_get_cached_path):
            with patch("ZebrafishEmbryoAnalyzerCore.length.load_model"):
                with pytest.raises(ModelNotCachedError):
                    logic.preload_models(
                        {"curvature": False, "eyes": False}
                    )
    finally:
        logic._MODEL_CACHE.clear()
        logic._MODEL_CACHE.update(saved_cache)
