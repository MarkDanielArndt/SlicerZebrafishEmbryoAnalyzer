"""Contract tests for the zebrafish-measure integration: Core shims and the
arguments logic.py hands to the library."""
import subprocess
import sys

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

SHIM_NAMES = {
    "length": ["compute_eye_metrics", "compute_tube_metrics", "compute_eye_diameters",
               "tube_length_border2border", "normalize_images", "apply_mask",
               "select_torch_device", "classification_curvature", "load_model",
               "plot_edges_with_curvature", "compute_curvature_profile",
               "compute_curvature", "preprocess_masked_image"],
    "seg": ["_UNET_CACHE", "_load_unet_model", "segmentation_pipeline"],
    "seg_helper": ["load_images_from_path", "segment_fish", "fill_holes", "grow_mask"],
    "scalebar": ["detect_scalebar", "draw_scalebar_endpoints", "calibrate_from_endpoints"],
}


@pytest.mark.parametrize("module,name", [(m, n) for m, ns in SHIM_NAMES.items() for n in ns])
def test_shim_resolves_name_from_library(module, name):
    import importlib
    shim = importlib.import_module(f"ZebrafishEmbryoAnalyzerCore.{module}")
    lib = importlib.import_module(f"zebrafish_measure.{module}")
    assert getattr(shim, name) is getattr(lib, name)
    assert name in dir(shim)


def test_shim_unknown_name_raises_attribute_error():
    import ZebrafishEmbryoAnalyzerCore.seg as shim
    with pytest.raises(AttributeError):
        shim.does_not_exist


def test_shim_mock_patch_applies_and_restores():
    import ZebrafishEmbryoAnalyzerCore.length as shim
    import zebrafish_measure.length as lib
    original = lib.load_model
    with patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as m:
        assert shim.load_model is m
    assert shim.load_model is original
    assert "load_model" not in vars(shim)


@pytest.mark.parametrize("module", list(SHIM_NAMES))
def test_importing_shim_is_lazy(module):
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import ZebrafishEmbryoAnalyzerCore.%s\n"
        "bad = [m for m in ('zebrafish_measure', 'cv2', 'torch', 'timm', "
        "'segmentation_models_pytorch', 'huggingface_hub', 'matplotlib') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    ) % (os.path.join(root, "ZebrafishEmbryoAnalyzer"), module)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "BAD:", out.stdout + out.stderr


# ---------------------------------------------------------------------------
# logic.py -> segmentation_pipeline kwargs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", ["fast", "general"])
def test_analyse_images_passes_explicit_pipeline_kwargs(
    tmp_path, synthetic_fish_image, mock_model_paths, model_id
):
    """Every enabled structure gets explicit encoder + model_type from the manifest
    (so the library's Segformer defaults are never used), local_only=True, CPU, and
    swim bladder post-processing disabled (Core behaviour)."""
    import cv2
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_SETS

    ms = MODEL_SETS[model_id]
    img_path = str(tmp_path / "fish.png")
    cv2.imwrite(img_path, synthetic_fish_image)
    dummy = np.zeros((256, 256), dtype=np.uint8)
    use_edema = "edema" in ms

    with patch("ZebrafishEmbryoAnalyzerCore.seg.segmentation_pipeline") as mock_pipeline, \
         patch("ZebrafishEmbryoAnalyzerCore.length.load_model") as mock_load:
        n_extra = 2 + (1 if use_edema else 0)
        mock_pipeline.return_value = tuple(
            [[synthetic_fish_image[:, :, ::-1]], [dummy], [dummy.copy()]]
            + [[dummy.copy()] for _ in range(n_extra)])
        mock_load.return_value = MagicMock()
        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        analyse_images(
            [img_path],
            {"length": False, "curvature": False, "ratio": False,
             "eyes": True, "edema": use_edema, "swimbladder": True, "hitl": False,
             "threshold": 0.85, "um_per_px": 22.99, "model_id": model_id},
        )

    kw = mock_pipeline.call_args.kwargs
    assert kw["local_only"] is True
    assert kw["device"] == "cpu"
    assert kw["swimbladder_biggest_only"] is False
    assert kw["swimbladder_boundary_offset_px"] is None
    for role, prefix in (("body", "body"), ("eye", "eye"), ("edema", "edema"),
                         ("swimbladder", "swimbladder")):
        if role == "edema" and not use_edema:
            assert "edema_encoder_name" not in kw
            assert "edema_model_filename" not in kw
            continue
        entry = ms[role]
        assert kw[f"{prefix}_encoder_name"] == entry["encoder"]
        assert kw[f"{prefix}_model_filename"] == entry["filename"]
        assert kw[f"{prefix}_model_type"] == entry.get("model_type", "Unet")
        if role != "body":
            assert kw[f"{prefix}_model_path"]
    assert kw["body_model_path"]


# ---------------------------------------------------------------------------
# _cached_load_unet
# ---------------------------------------------------------------------------

def test_cached_load_unet_forwards_local_only_and_device_and_caches(monkeypatch):
    from ZebrafishEmbryoAnalyzerLib import logic
    orig = MagicMock(side_effect=lambda **kw: object())
    monkeypatch.setattr(logic, "_original_load_unet", orig)
    monkeypatch.setattr(logic, "_MODEL_CACHE", {})

    m1 = logic._cached_load_unet(model_path="/x.pth", label="body model",
                                 encoder_name="vgg19", model_type="Unet",
                                 local_only=True, device="cpu")
    m2 = logic._cached_load_unet(model_path="/x.pth", label="body model",
                                 encoder_name="vgg19", model_type="Unet",
                                 local_only=True, device="cpu")
    assert m1 is m2
    assert orig.call_count == 1
    kw = orig.call_args.kwargs
    assert kw["local_only"] is True and kw["device"] == "cpu"
    assert kw["model_path"] == "/x.pth" and kw["encoder_name"] == "vgg19"

    # different device / model_type / encoder -> separate cache entries
    m3 = logic._cached_load_unet(model_path="/x.pth", encoder_name="vgg19",
                                 model_type="Unet", local_only=True, device="cuda")
    m4 = logic._cached_load_unet(model_path="/x.pth", encoder_name="vgg19",
                                 model_type="FPN", local_only=True, device="cpu")
    assert len({id(m1), id(m3), id(m4)}) == 3
    assert orig.call_count == 3


def test_install_model_cache_patches_library_module(monkeypatch):
    import zebrafish_measure.seg as lib_seg
    from ZebrafishEmbryoAnalyzerLib import logic
    monkeypatch.setattr(logic, "_original_load_unet", None)
    saved = lib_seg._load_unet_model
    try:
        logic._install_model_cache()
        assert lib_seg._load_unet_model is logic._cached_load_unet
        assert logic._original_load_unet is lib_seg._load_unet_model_original
    finally:
        lib_seg._load_unet_model = saved
