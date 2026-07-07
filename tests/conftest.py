import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Slicer puts the module directory on sys.path at runtime; mirror that here so
# `ZebrafishEmbryoAnalyzerCore` and `ZebrafishEmbryoAnalyzerLib` import the same way in tests.
_MODULE_DIR = Path(__file__).resolve().parent.parent / "ZebrafishEmbryoAnalyzer"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


@pytest.fixture
def synthetic_fish_image():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.ellipse(img, (128, 128), (100, 30), 0, 0, 360, (200, 200, 200), -1)
    return img


@pytest.fixture
def synthetic_fish_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    import cv2
    cv2.ellipse(mask, (128, 128), (100, 30), 0, 0, 360, 255, -1)
    return mask


@pytest.fixture
def mock_model_paths(tmp_path):
    """Patch get_cached_path so existence checks in logic.py pass without real models.

    Returns a dict mapping entry id -> Path so tests can inspect paths if needed.
    """
    created: dict = {}

    def fake_get_cached_path(entry):
        p = tmp_path / entry["filename"]
        if not p.exists():
            p.write_bytes(b"dummy weights")
        created[entry["id"]] = p
        return p

    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest.get_cached_path",
               side_effect=fake_get_cached_path):
        yield created
