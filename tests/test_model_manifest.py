"""
Tests for ZebrafishAnalysisLib.model_manifest.

Pure-Python module: no slicer, qt, vtk, ctk, or torch required.
"""

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ZebrafishAnalysisLib.model_manifest import (
    MODELS,
    MODEL_SETS,
    _CACHE_DIR,
    get_cached_path,
    get_missing_models,
    verify_checksum,
)

MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishAnalysis"
)


# ---------------------------------------------------------------------------
# AC1: model_manifest importable without slicer / qt / torch / vtk
# ---------------------------------------------------------------------------

def test_manifest_importable_without_heavy_deps():
    """Importing model_manifest must not pull in torch, slicer, qt, or vtk."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            sys.modules["torch"] = None
            sys.modules["slicer"] = None
            sys.modules["qt"] = None
            sys.modules["vtk"] = None
            sys.modules["ctk"] = None
            from ZebrafishAnalysisLib import model_manifest
            print("OK")
        """)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": MODULE_DIR},
    )
    assert result.returncode == 0, (
        "model_manifest must be importable without torch/slicer/qt/vtk/ctk.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# AC2: MODELS dict has required fields for all entries
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("id", "repo_id", "filename", "revision", "label",
                    "sha256", "size_bytes", "license", "preprocessing_compat")


@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_model_entry_has_required_fields(model_id, entry):
    for field in _REQUIRED_FIELDS:
        assert field in entry, f"MODELS[{model_id!r}] missing field {field!r}"


def test_models_has_all_expected_ids():
    expected = {"general_body", "general_eye", "general_edema", "curvature", "desy_body", "desy_eye"}
    assert expected.issubset(set(MODELS.keys()))


def test_model_sets_has_general_and_desy():
    assert "general" in MODEL_SETS
    assert "desy" in MODEL_SETS


@pytest.mark.parametrize("variant", ["general", "desy"])
def test_model_set_has_all_roles(variant):
    required_roles = {"body", "eye", "curvature"}
    assert required_roles.issubset(set(MODEL_SETS[variant].keys()))
    # edema is intentionally excluded from MODEL_SETS to avoid gating the Run
    # button on a model that is not yet used (include_edema defaults to False).
    assert "edema" not in MODEL_SETS[variant]


# ---------------------------------------------------------------------------
# AC3: get_cached_path returns consistent Path objects
# ---------------------------------------------------------------------------

def test_get_cached_path_returns_path_object():
    entry = MODELS["general_body"]
    p = get_cached_path(entry)
    assert isinstance(p, Path)


def test_get_cached_path_uses_cache_dir():
    entry = MODELS["general_body"]
    p = get_cached_path(entry)
    assert p.parent == _CACHE_DIR


def test_get_cached_path_uses_filename():
    entry = MODELS["general_body"]
    p = get_cached_path(entry)
    assert p.name == entry["filename"]


def test_get_cached_path_consistent():
    """Same entry → same path on repeated calls."""
    entry = MODELS["curvature"]
    assert get_cached_path(entry) == get_cached_path(entry)


def test_different_entries_different_paths():
    """Different model entries produce different cache paths."""
    p1 = get_cached_path(MODELS["general_body"])
    p2 = get_cached_path(MODELS["general_eye"])
    assert p1 != p2


# ---------------------------------------------------------------------------
# AC4: verify_checksum — PENDING always returns True
# ---------------------------------------------------------------------------

def test_verify_checksum_pending_returns_true_no_file():
    """PENDING sha256 must return True even when file does not exist."""
    result = verify_checksum("/nonexistent/path/model.pth", "PENDING")
    assert result is True


def test_verify_checksum_pending_returns_true_with_file(tmp_path):
    f = tmp_path / "model.pth"
    f.write_bytes(b"dummy weights")
    assert verify_checksum(str(f), "PENDING") is True


# ---------------------------------------------------------------------------
# AC5: verify_checksum — actual hex matches / mismatches correctly
# ---------------------------------------------------------------------------

def test_verify_checksum_correct_hash(tmp_path):
    data = b"zebrafish model weights"
    f = tmp_path / "model.pth"
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert verify_checksum(str(f), expected) is True


def test_verify_checksum_wrong_hash(tmp_path):
    data = b"zebrafish model weights"
    f = tmp_path / "model.pth"
    f.write_bytes(data)
    assert verify_checksum(str(f), "a" * 64) is False


def test_verify_checksum_missing_file_returns_false():
    result = verify_checksum("/nonexistent/path/model.pth", "a" * 64)
    assert result is False


# ---------------------------------------------------------------------------
# AC6: get_missing_models — returns only entries without valid cached files
# ---------------------------------------------------------------------------

def test_get_missing_models_all_absent(tmp_path):
    """When no files are cached, all entries are returned as missing."""
    with patch("ZebrafishAnalysisLib.model_manifest._CACHE_DIR", tmp_path):
        # Reload get_missing_models to pick up patched _CACHE_DIR.
        from ZebrafishAnalysisLib.model_manifest import MODELS as _MODELS

        # Build a small model_set using real entries but pointing at tmp_path.
        model_set = {
            "body": _MODELS["general_body"],
            "curvature": _MODELS["curvature"],
        }
        # get_cached_path uses module-level _CACHE_DIR; patch it.
        import ZebrafishAnalysisLib.model_manifest as _mm
        orig = _mm._CACHE_DIR
        _mm._CACHE_DIR = tmp_path
        try:
            missing = get_missing_models(model_set)
        finally:
            _mm._CACHE_DIR = orig

    assert len(missing) == 2


def test_get_missing_models_none_absent(tmp_path):
    """When all files are present and non-empty, nothing is reported missing."""
    import ZebrafishAnalysisLib.model_manifest as _mm
    from ZebrafishAnalysisLib.model_manifest import MODELS as _MODELS

    model_set = {
        "body": _MODELS["general_body"],
        "curvature": _MODELS["curvature"],
    }

    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        # Write dummy files for each entry.
        for entry in model_set.values():
            p = tmp_path / entry["filename"]
            p.write_bytes(b"dummy weights")
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert missing == []


def test_get_missing_models_empty_file_counts_as_missing(tmp_path):
    """A zero-byte file is treated as missing."""
    import ZebrafishAnalysisLib.model_manifest as _mm
    from ZebrafishAnalysisLib.model_manifest import MODELS as _MODELS

    model_set = {"body": _MODELS["general_body"]}
    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        p = tmp_path / _MODELS["general_body"]["filename"]
        p.write_bytes(b"")  # empty
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert len(missing) == 1


def test_get_missing_models_partial(tmp_path):
    """Only entries that are absent are returned."""
    import ZebrafishAnalysisLib.model_manifest as _mm
    from ZebrafishAnalysisLib.model_manifest import MODELS as _MODELS

    model_set = {
        "body": _MODELS["general_body"],
        "curvature": _MODELS["curvature"],
    }
    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        # Write only the body file.
        p = tmp_path / _MODELS["general_body"]["filename"]
        p.write_bytes(b"dummy weights")
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert len(missing) == 1
    assert missing[0]["id"] == "curvature"


# ---------------------------------------------------------------------------
# DESY variant: curvature reuses general entry; edema only in MODELS, not MODEL_SETS
# ---------------------------------------------------------------------------

def test_desy_reuses_curvature():
    assert MODEL_SETS["desy"]["curvature"] is MODEL_SETS["general"]["curvature"]


def test_edema_in_models_but_not_in_model_sets():
    """Edema model described in MODELS but excluded from MODEL_SETS (no UI control)."""
    assert "general_edema" in MODELS
    assert "edema" not in MODEL_SETS["general"]
    assert "edema" not in MODEL_SETS["desy"]


# ---------------------------------------------------------------------------
# size_bytes: approximate pre-download size estimates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_size_bytes_is_positive_int(model_id, entry):
    """size_bytes must be a positive integer for all known models."""
    sb = entry["size_bytes"]
    assert isinstance(sb, int), f"MODELS[{model_id!r}]['size_bytes'] is not int"
    assert sb > 0, f"MODELS[{model_id!r}]['size_bytes'] must be > 0"


def test_curvature_smaller_than_segmentation_models():
    """Curvature model is ~340 MB; segmentation models are ~530 MB."""
    assert MODELS["curvature"]["size_bytes"] < MODELS["general_body"]["size_bytes"]


def test_size_bytes_reasonable_range():
    """All size estimates fall within 100 MB – 2 GB (sanity bounds)."""
    for model_id, entry in MODELS.items():
        sb = entry["size_bytes"]
        assert 100_000_000 <= sb <= 2_000_000_000, (
            f"MODELS[{model_id!r}]['size_bytes']={sb} outside plausible range"
        )
