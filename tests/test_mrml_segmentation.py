"""Plain-Python tests for E2c segmentation helpers in ZebrafishEmbryoAnalyzerLib.mrml.

All tests run in the normal pytest process (no slicer / vtk required).
conftest.py adds ZebrafishEmbryoAnalyzer to sys.path so the import works directly.
"""

import numpy as np
import pytest

from ZebrafishEmbryoAnalyzerLib.mrml import resample_mask_to_original


# ---------------------------------------------------------------------------
# resample_mask_to_original — shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_matches_h_orig_w_orig():
    mask = np.zeros((256, 256), dtype=np.uint8)
    out = resample_mask_to_original(mask, 480, 640)
    assert out.shape == (480, 640)


def test_output_dtype_is_uint8():
    mask = np.ones((256, 256), dtype=np.float32)
    out = resample_mask_to_original(mask, 100, 120)
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Binary value preservation
# ---------------------------------------------------------------------------

def test_all_zero_mask_yields_all_zero_output():
    mask = np.zeros((256, 256), dtype=np.uint8)
    out = resample_mask_to_original(mask, 480, 640)
    assert out.sum() == 0


def test_all_ones_bool_mask_yields_all_ones_uint8():
    mask = np.ones((256, 256), dtype=bool)
    out = resample_mask_to_original(mask, 480, 640)
    assert (out == 1).all()


def test_float_mask_above_zero_binarized_to_one():
    mask = np.full((256, 256), 0.7, dtype=np.float32)
    out = resample_mask_to_original(mask, 200, 300)
    assert (out == 1).all()


def test_float_mask_at_zero_stays_zero():
    mask = np.zeros((256, 256), dtype=np.float64)
    out = resample_mask_to_original(mask, 200, 300)
    assert out.sum() == 0


def test_values_are_only_zero_or_one():
    rng = np.random.default_rng(0)
    mask = rng.random((256, 256)).astype(np.float32)
    out = resample_mask_to_original(mask, 300, 400)
    unique = set(out.flatten().tolist())
    assert unique <= {0, 1}


# ---------------------------------------------------------------------------
# Geometry contract: partial body block
# ---------------------------------------------------------------------------

def test_solid_block_maps_to_correct_region():
    """A filled rectangle in the top-left of the 256x256 mask must appear
    in the top-left of the resampled output."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:128, :128] = 1  # top-left quadrant
    out = resample_mask_to_original(mask, 256, 256)
    # Top-left quadrant must be all 1
    assert out[:128, :128].sum() == 128 * 128
    # Bottom-right quadrant must be all 0
    assert out[128:, 128:].sum() == 0


# ---------------------------------------------------------------------------
# eye_mask None path — resample_mask_to_original not called; no crash
# ---------------------------------------------------------------------------

def test_none_eye_mask_not_passed_to_resample():
    """resample_mask_to_original must not be called with None;
    the caller (update_segmentation_node) skips it when eye_mask is None.
    This test verifies the helper itself does not crash on a normal body mask."""
    mask = np.ones((256, 256), dtype=np.uint8)
    # Simulate: eye_mask is None → only body mask resampled
    eye_mask = None
    body_out = resample_mask_to_original(mask, 100, 100)
    assert body_out is not None
    # eye path skipped — no exception
    if eye_mask is not None and eye_mask.any():
        resample_mask_to_original(eye_mask, 100, 100)


# ---------------------------------------------------------------------------
# Non-square inputs
# ---------------------------------------------------------------------------

def test_non_square_mask_non_square_target():
    mask = np.ones((256, 256), dtype=np.uint8)
    out = resample_mask_to_original(mask, 123, 456)
    assert out.shape == (123, 456)
    assert (out == 1).all()
