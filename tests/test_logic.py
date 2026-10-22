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


def test_analyse_images_computes_edema_area(tmp_path, synthetic_fish_image, mock_model_paths):
    """With edema requested on the Fast model, the returned result carries the mask
    and a non-zero physical area computed from it."""
    import cv2
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_grown = dummy_mask.copy()
    dummy_edema = np.zeros((256, 256), dtype=np.uint8)
    dummy_edema[10:20, 10:20] = 255  # 100 px^2 of edema

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as mock_load:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_grown],
            [dummy_edema],
        )
        mock_load.return_value = MagicMock()

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        results = analyse_images(
            [img_path],
            {"length": False, "curvature": False, "ratio": False,
             "eyes": False, "edema": True, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99, "model_id": "fast"},
        )

    assert len(results) == 1
    assert results[0]["edema_mask"] is not None
    assert results[0]["edema_area"] is not None
    assert results[0]["edema_area"] > 0


@pytest.mark.parametrize("model_id,expected_size,expected_type", [
    ("fast",    (256, 256), "Segformer"),
    ("general", (512, 512), "Segformer"),
])
def test_analyse_images_threads_preset_resolution_and_architecture(
    tmp_path, synthetic_fish_image, mock_model_paths, model_id, expected_size, expected_type
):
    """Each preset must reach segmentation_pipeline with its own target_size and
    swim bladder architecture. Getting either wrong is silent: a 512px-trained net
    fed 256px input, or an FPN checkpoint loaded into a Unet, both "work" and just
    produce garbage masks."""
    import cv2
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as mock_load:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_mask.copy()],
            [dummy_mask.copy()],
        )
        mock_load.return_value = MagicMock()

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        analyse_images(
            [img_path],
            {"length": False, "curvature": False, "ratio": False,
             "eyes": False, "swimbladder": True, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99, "model_id": model_id},
        )

    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["target_size"] == expected_size
    assert kwargs["swimbladder_model_type"] == expected_type


def test_analyse_images_computes_swim_bladder_area_and_width(tmp_path, synthetic_fish_image, mock_model_paths):
    """With swim bladder requested, the returned result carries the mask and
    non-zero area/width computed via compute_tube_metrics (min-area rect)."""
    import cv2
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_grown = dummy_mask.copy()
    dummy_swim = np.zeros((256, 256), dtype=np.uint8)
    dummy_swim[100:110, 50:150] = 255  # 10x100 rectangle

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as mock_load:

        mock_pipeline.return_value = (
            [synthetic_fish_image[:, :, ::-1]],
            [dummy_mask],
            [dummy_grown],
            [dummy_swim],
        )
        mock_load.return_value = MagicMock()

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        results = analyse_images(
            [img_path],
            {"length": False, "curvature": False, "ratio": False,
             "eyes": False, "swimbladder": True, "hitl": False, "threshold": 0.85,
             "um_per_px": 22.99, "model_id": "general"},
        )

    assert len(results) == 1
    assert results[0]["swimbladder_mask"] is not None
    assert results[0]["swim_area"] is not None
    assert results[0]["swim_area"] > 0
    assert results[0]["swim_width"] is not None
    assert results[0]["swim_width"] > 0


def test_analyse_images_raises_model_not_cached_when_swimbladder_missing(tmp_path, synthetic_fish_image):
    """A missing swim bladder model file must raise ModelNotCachedError."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    def fake_get_cached_path(entry):
        return tmp_path / entry["filename"]

    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
               side_effect=fake_get_cached_path):
        with pytest.raises(ModelNotCachedError):
            analyse_images(
                [img_path],
                {"length": False, "curvature": False, "ratio": False,
                 "eyes": False, "swimbladder": True, "hitl": False, "threshold": 0.85,
                 "um_per_px": 22.99, "model_id": "general"},
            )


@pytest.mark.parametrize("entry_point", ["analyse_images", "preload_models"])
def test_edema_on_preset_without_edema_model_raises_clearly(tmp_path, synthetic_fish_image, entry_point, monkeypatch):
    """A request asking for edema under a preset that has none must fail with an
    actionable message, not a bare KeyError and not a silently-empty column.

    Only reachable through an inconsistent request (the widget greys the box
    out), but a scene saved before the role was withdrawn is exactly that.
    Both shipped presets now have an edema model, so the missing role is
    simulated by removing it from the "general" set."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib import logic
    from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_SETS
    monkeypatch.setitem(MODEL_SETS, "general",
                        {k: v for k, v in MODEL_SETS["general"].items() if k != "edema"})

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    params = {"length": False, "curvature": False, "ratio": False,
              "eyes": False, "edema": True, "hitl": False, "threshold": 0.85,
              "um_per_px": 22.99, "model_id": "general"}

    with pytest.raises(AnalysisInputError, match="not available"):
        if entry_point == "analyse_images":
            logic.analyse_images([img_path], params)
        else:
            logic.preload_models(params)


def test_analyse_images_raises_model_not_cached_when_edema_missing(tmp_path, synthetic_fish_image):
    """A missing edema model file must raise ModelNotCachedError, not silently proceed."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    def fake_get_cached_path(entry):
        return tmp_path / entry["filename"]

    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
               side_effect=fake_get_cached_path):
        with pytest.raises(ModelNotCachedError):
            analyse_images(
                [img_path],
                {"length": False, "curvature": False, "ratio": False,
                 "eyes": False, "edema": True, "hitl": False, "threshold": 0.85,
                 "um_per_px": 22.99, "model_id": "fast"},
            )


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
    """Files present and matching their pinned checksum count as cached."""
    from ZebrafishEmbryoAnalyzerLib.model_manifest import get_missing_models, MODEL_SETS
    import hashlib
    data = b"\x00"
    digest = hashlib.sha256(data).hexdigest()
    model_set = {role: {**entry, "sha256": digest} for role, entry in MODEL_SETS["general"].items()}
    for entry in model_set.values():
        f = tmp_path / entry["filename"]
        f.write_bytes(data)
    with patch(
        "ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
        side_effect=lambda entry: tmp_path / entry["filename"],
    ):
        result = get_missing_models(model_set)
    assert result == []


def test_models_to_download_corrupted_counts_as_missing(tmp_path):
    """A present, non-empty file whose content does not match the pinned checksum
    (e.g. a truncated/interrupted download) must count as missing, not cached —
    otherwise it looks available until torch.load() fails deep inside analysis."""
    from ZebrafishEmbryoAnalyzerLib.model_manifest import get_missing_models, MODEL_SETS
    model_set = MODEL_SETS["general"]
    for entry in model_set.values():
        f = tmp_path / entry["filename"]
        f.write_bytes(b"truncated download, wrong content")  # real sha256 won't match
    with patch(
        "ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
        side_effect=lambda entry: tmp_path / entry["filename"],
    ):
        result = get_missing_models(model_set)
    assert len(result) == len(model_set)


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


def test_analyse_images_raises_model_not_cached_when_body_corrupted(tmp_path, synthetic_fish_image):
    """A present but checksum-mismatched body model file (truncated/interrupted
    download) must raise ModelNotCachedError, the same as a genuinely missing file —
    this is the real-world failure mode: torch.load() otherwise fails deep inside
    analysis with a cryptic pickle error instead of the normal download-prompt flow."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError

    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)

    def fake_get_cached_path(entry):
        p = tmp_path / entry["filename"]
        p.write_bytes(b"truncated download, wrong content")
        return p

    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
               side_effect=fake_get_cached_path):
        with pytest.raises(ModelNotCachedError):
            analyse_images(
                [img_path],
                {"length": True, "curvature": False, "ratio": False,
                 "eyes": False, "hitl": False, "threshold": 0.85, "um_per_px": 22.99},
            )


def test_preload_models_raises_model_not_cached_when_body_corrupted(tmp_path):
    """Same as the missing-file case above, but for a present, checksum-mismatched
    file — must not be treated as usable."""
    from ZebrafishEmbryoAnalyzerLib import logic
    from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError

    def fake_get_cached_path(entry):
        p = tmp_path / entry["filename"]
        p.write_bytes(b"truncated download, wrong content")
        return p

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
