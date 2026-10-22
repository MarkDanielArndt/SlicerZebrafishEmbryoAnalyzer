"""
Tests for ZebrafishEmbryoAnalyzerLib.model_manifest.

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

from ZebrafishEmbryoAnalyzerLib.model_manifest import (
    MODELS,
    MODEL_SETS,
    _CACHE_DIR,
    _default_cache_dir,
    checksum_mismatch_error,
    get_cached_path,
    get_missing_models,
    verify_checksum,
)

MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
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
            from ZebrafishEmbryoAnalyzerLib import model_manifest
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
    expected = {f"{p}_{r}" for p in ("fast", "general")
                for r in ("body", "eye", "edema", "swimbladder")} | {"curvature"}
    assert set(MODELS.keys()) == expected


def test_model_sets_has_all_presets():
    assert set(MODEL_SETS.keys()) == {"fast", "general"}


def test_no_desy_preset_or_models():
    """DESY is webapp-only: no preset, model entry or target size may remain."""
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_TARGET_SIZE
    assert "desy" not in MODEL_SETS
    assert "desy" not in MODEL_TARGET_SIZE
    assert not [k for k in MODELS if "desy" in k]


@pytest.mark.parametrize("variant", ["fast", "general"])
def test_model_set_has_every_always_available_role(variant):
    """Body, eye, curvature, edema and swim bladder exist for every preset — the
    webapp gates none of them, and a resolution-matched model exists for each."""
    always = {"body", "eye", "curvature", "edema", "swimbladder"}
    assert always.issubset(set(MODEL_SETS[variant].keys()))


def test_edema_offered_on_both_presets_with_matching_resolution():
    """Both presets have their own resolution-matched edema model (256px / 512px)."""
    assert MODEL_SETS["fast"]["edema"] is MODELS["fast_edema"]
    assert MODEL_SETS["general"]["edema"] is MODELS["general_edema"]
    assert "_256_" in MODELS["fast_edema"]["filename"]
    assert "_512_" in MODELS["general_edema"]["filename"]


def test_fast_body_is_distinct_from_general_body():
    """fast_body (256px) must not be confused with general_body (512px) —
    same underlying webapp repo, genuinely different files/resolutions."""
    fast = MODELS["fast_body"]
    general = MODELS["general_body"]
    assert fast["filename"] != general["filename"]
    assert fast["sha256"] != general["sha256"]


def test_model_target_size_covers_every_preset():
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_TARGET_SIZE
    assert set(MODEL_TARGET_SIZE.keys()) == set(MODEL_SETS.keys())
    assert MODEL_TARGET_SIZE["fast"] == (256, 256)
    assert MODEL_TARGET_SIZE["general"] == (512, 512)


@pytest.mark.parametrize("variant", ["fast", "general"])
def test_every_segmentation_role_is_segformer_mit_b3(variant):
    """Both presets use the webapp's Segformer mit_b3 models for every
    segmentation role. seg.py's _load_unet_model branches on model_type, so
    getting this wrong silently builds the wrong net."""
    for role in ("body", "eye", "edema", "swimbladder"):
        entry = MODEL_SETS[variant][role]
        assert entry["model_type"] == "Segformer", (variant, role)
        assert entry["encoder"] == "mit_b3", (variant, role)
        assert entry["revision"] == "993a5d31cb3dde4ac8af72a5cbee389e1c25f239"
        assert entry["repo_id"] == "markdanielarndt/Zebrafish_Segmentation"


_PINNED = {
    "fast_body": ("best_model_body_256_segformer_mit_b3.pth", 178629809,
                  "e4f04e97803a665839c75c70bdc75a3a89dbee28e72da1b640daf4ed2e2a2677"),
    "general_body": ("best_model_body_512_segformer_mit_b3.pth", 178629809,
                     "54651c627195bb7291a65fe0808992565ff74b222925efc4c810ed637074e8fc"),
    "fast_edema": ("best_model_edema_256_segformer_mit_b3.pth", 178630403,
                   "58ed666bb6540590ccade64027ee25fbde23bc950b6e5803a7bb8eccdecf043c"),
    "general_edema": ("best_model_edema_512_segformer_mit_b3.pth", 178630403,
                      "8f8f00305c2367d54dd114d0a316bdd4ad3251dae18f948cde648dbb48267ac0"),
    "fast_eye": ("best_model_eye_256_segformer_mit_b3.pth", 178629215,
                 "5bf80e5bb92912200614cb73d9ce3ff2c811d2441c8660c4336f6f8713a935e7"),
    "general_eye": ("best_model_eye_512_segformer_mit_b3.pth", 178629215,
                    "646fd95ac5dc6ecad6d4320c4349648e796b75e50b113a8f4efd023c108592e0"),
    # "swimmbladder" (double m) is the real HF filename.
    "fast_swimbladder": ("best_model_swimmbladder_256_segformer_mit_b3.pth", 178634561,
                         "8e62d653da75a996eb3802465f29ba3f2d7f5255c5411954583d143f91f184dd"),
    "general_swimbladder": ("best_model_swimmbladder_512_segformer_mit_b3.pth", 178634561,
                            "dd785146ec5125a4bb1a038b4ef5e8a25c99076b1c4fb007d68c85b89ea1942a"),
}


@pytest.mark.parametrize("model_id", sorted(_PINNED))
def test_segmentation_entries_pinned_to_hf_filename_size_and_hash(model_id):
    filename, size, sha = _PINNED[model_id]
    entry = MODELS[model_id]
    assert entry["filename"] == filename
    assert entry["size_bytes"] == size
    assert entry["sha256"] == sha


def test_swimbladder_entries_distinct_fast_vs_general():
    fast = MODELS["fast_swimbladder"]
    general = MODELS["general_swimbladder"]
    assert fast["filename"] != general["filename"]
    assert fast["sha256"] != general["sha256"]


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
# AC4: verify_checksum — PENDING raises ValueError (BLK-05)
# ---------------------------------------------------------------------------

def test_verify_checksum_pending_raises_value_error_no_file():
    """PENDING sha256 must raise ValueError (configuration error)."""
    with pytest.raises(ValueError, match="placeholder or missing"):
        verify_checksum("/nonexistent/path/model.pth", "PENDING")


def test_verify_checksum_pending_raises_value_error_with_file(tmp_path):
    """PENDING sha256 raises ValueError even when file exists."""
    f = tmp_path / "model.pth"
    f.write_bytes(b"dummy weights")
    with pytest.raises(ValueError, match="placeholder or missing"):
        verify_checksum(str(f), "PENDING")


def test_verify_checksum_empty_sha256_raises_value_error(tmp_path):
    """Empty sha256 string raises ValueError."""
    f = tmp_path / "model.pth"
    f.write_bytes(b"dummy weights")
    with pytest.raises(ValueError, match="placeholder or missing"):
        verify_checksum(str(f), "")


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
    with patch("ZebrafishEmbryoAnalyzerLib.model_manifest._CACHE_DIR", tmp_path):
        # Reload get_missing_models to pick up patched _CACHE_DIR.
        from ZebrafishEmbryoAnalyzerLib.model_manifest import MODELS as _MODELS

        # Build a small model_set using real entries but pointing at tmp_path.
        model_set = {
            "body": _MODELS["general_body"],
            "curvature": _MODELS["curvature"],
        }
        # get_cached_path uses module-level _CACHE_DIR; patch it.
        import ZebrafishEmbryoAnalyzerLib.model_manifest as _mm
        orig = _mm._CACHE_DIR
        _mm._CACHE_DIR = tmp_path
        try:
            missing = get_missing_models(model_set)
        finally:
            _mm._CACHE_DIR = orig

    assert len(missing) == 2


def test_get_missing_models_none_absent(tmp_path):
    """When all files are present and match their pinned checksum, nothing is missing."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as _mm
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODELS as _MODELS

    data = b"dummy weights"
    digest = hashlib.sha256(data).hexdigest()
    model_set = {
        "body": {**_MODELS["general_body"], "sha256": digest},
        "curvature": {**_MODELS["curvature"], "sha256": digest},
    }

    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        for entry in model_set.values():
            p = tmp_path / entry["filename"]
            p.write_bytes(data)
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert missing == []


def test_get_missing_models_empty_file_counts_as_missing(tmp_path):
    """A zero-byte file is treated as missing."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as _mm
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODELS as _MODELS

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


def test_get_missing_models_corrupted_file_counts_as_missing(tmp_path):
    """A present, non-empty file whose content does not match the pinned checksum
    (e.g. a truncated/interrupted download) must count as missing — this is the
    real-world failure mode: it otherwise looks cached until torch.load() fails
    deep inside analysis with a cryptic pickle error."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as _mm
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODELS as _MODELS

    model_set = {"body": _MODELS["general_body"]}
    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        p = tmp_path / _MODELS["general_body"]["filename"]
        p.write_bytes(b"truncated download, wrong content")  # real sha256 won't match
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert len(missing) == 1


def test_get_missing_models_partial(tmp_path):
    """Only entries that are absent (or corrupted) are returned."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as _mm
    from ZebrafishEmbryoAnalyzerLib.model_manifest import MODELS as _MODELS

    data = b"dummy weights"
    digest = hashlib.sha256(data).hexdigest()
    model_set = {
        "body": {**_MODELS["general_body"], "sha256": digest},
        "curvature": _MODELS["curvature"],
    }
    orig = _mm._CACHE_DIR
    _mm._CACHE_DIR = tmp_path
    try:
        # Write only the body file, with matching checksum.
        p = tmp_path / model_set["body"]["filename"]
        p.write_bytes(data)
        missing = get_missing_models(model_set)
    finally:
        _mm._CACHE_DIR = orig

    assert len(missing) == 1
    assert missing[0]["id"] == "curvature"


def test_no_unwired_model_entries():
    """Every MODELS entry must be reachable from some preset. An entry that is
    described but wired nowhere previously caused it to be mistaken for dead
    weight and mis-documented; collect_all_model_entries() also drives the
    Settings tab's cache list, so an orphan would be invisible there."""
    wired = {e["id"] for variant in MODEL_SETS.values() for e in variant.values()}
    assert set(MODELS.keys()) == wired


# ---------------------------------------------------------------------------
# size_bytes: approximate pre-download size estimates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_size_bytes_is_positive_int(model_id, entry):
    """size_bytes must be a positive integer for all known models."""
    sb = entry["size_bytes"]
    assert isinstance(sb, int), f"MODELS[{model_id!r}]['size_bytes'] is not int"
    assert sb > 0, f"MODELS[{model_id!r}]['size_bytes'] must be > 0"


def test_curvature_larger_than_segmentation_models():
    """Curvature model (~352 MB) is larger than body (~116 MB) and eye (~95 MB) models."""
    assert MODELS["curvature"]["size_bytes"] > MODELS["general_body"]["size_bytes"]
    assert MODELS["curvature"]["size_bytes"] > MODELS["general_eye"]["size_bytes"]


def test_size_bytes_reasonable_range():
    """All size estimates fall within 50 MB – 2 GB (sanity bounds)."""
    for model_id, entry in MODELS.items():
        sb = entry["size_bytes"]
        assert 50_000_000 <= sb <= 2_000_000_000, (
            f"MODELS[{model_id!r}]['size_bytes']={sb} outside plausible range"
        )


# ---------------------------------------------------------------------------
# J1: Platform-independent cache path
# ---------------------------------------------------------------------------

def test_default_cache_dir_returns_path():
    """_default_cache_dir() must return a pathlib.Path object."""
    result = _default_cache_dir()
    assert isinstance(result, Path)


def test_default_cache_dir_is_absolute():
    """Cache directory must be an absolute path on all platforms."""
    result = _default_cache_dir()
    assert result.is_absolute(), f"Cache dir is not absolute: {result}"


def test_default_cache_dir_without_platformdirs(monkeypatch):
    """Falls back to ~/.cache/zebrafish_models when platformdirs is absent."""
    import sys
    import importlib

    # Simulate platformdirs being missing by blocking the import
    original = sys.modules.get("platformdirs", None)
    sys.modules["platformdirs"] = None  # causes ImportError on 'from platformdirs import ...'
    try:
        import ZebrafishEmbryoAnalyzerLib.model_manifest as mm
        result = mm._default_cache_dir()
    finally:
        if original is None:
            del sys.modules["platformdirs"]
        else:
            sys.modules["platformdirs"] = original

    expected = Path.home() / ".cache" / "zebrafish_models"
    assert result == expected


def test_default_cache_dir_platformdirs_raises_oserror(monkeypatch):
    """Falls back to ~/.cache/zebrafish_models when user_cache_dir raises OSError."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as mm

    def _raise_oserror(*args, **kwargs):
        raise OSError("restricted LOCALAPPDATA")

    monkeypatch.setattr("platformdirs.user_cache_dir", _raise_oserror)
    result = mm._default_cache_dir()
    expected = Path.home() / ".cache" / "zebrafish_models"
    assert result == expected


def test_cache_dir_injectable_via_module_attr(tmp_path):
    """get_cached_path respects a patched _CACHE_DIR (enables unit testing)."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as mm
    orig = mm._CACHE_DIR
    mm._CACHE_DIR = tmp_path
    try:
        p = mm.get_cached_path(MODELS["general_body"])
        assert p.parent == tmp_path
    finally:
        mm._CACHE_DIR = orig


def test_get_cached_path_no_string_concatenation():
    """get_cached_path must return a Path whose parts use OS separator, not '/'."""
    p = get_cached_path(MODELS["general_body"])
    # Path objects always use os.sep internally; confirm it's a Path not str
    assert isinstance(p, Path)
    # The name must equal the filename from the manifest
    assert p.name == MODELS["general_body"]["filename"]


# ---------------------------------------------------------------------------
# HIGH-01: All revisions must be immutable commit SHAs (not floating "main")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_revision_is_not_main(model_id, entry):
    """No MODELS entry may use the floating 'main' branch as revision."""
    revision = entry["revision"]
    assert revision != "main", (
        f"MODELS[{model_id!r}]['revision'] is still 'main' — use an immutable commit SHA."
    )


@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_revision_looks_like_commit_sha(model_id, entry):
    """Revision must look like a 40-character hex commit SHA."""
    revision = entry["revision"]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), (
        f"MODELS[{model_id!r}]['revision']={revision!r} is not a 40-char lowercase hex SHA."
    )


# ---------------------------------------------------------------------------
# BLK-05: All sha256 values are real 64-char hex strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_sha256_is_not_pending(model_id, entry):
    """No MODELS entry may have sha256='PENDING'."""
    sha = entry["sha256"]
    assert sha != "PENDING", (
        f"MODELS[{model_id!r}]['sha256'] is still 'PENDING' — update with real checksum."
    )


@pytest.mark.parametrize("model_id,entry", list(MODELS.items()))
def test_sha256_is_64_char_lowercase_hex(model_id, entry):
    """sha256 must be exactly 64 lowercase hex characters."""
    sha = entry.get("sha256", "")
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
        f"MODELS[{model_id!r}]['sha256']={sha!r} is not a 64-char lowercase hex string."
    )


# ---------------------------------------------------------------------------
# BLK-05: checksum_mismatch_error returns useful string
# ---------------------------------------------------------------------------

def test_checksum_mismatch_error_contains_expected_and_actual():
    entry = {"id": "general_body", "sha256": "a" * 64}
    msg = checksum_mismatch_error(entry, "/some/path/model.pth", "b" * 64)
    assert "general_body" in msg
    assert "a" * 64 in msg
    assert "b" * 64 in msg
    assert "re-download" in msg.lower()


def test_checksum_mismatch_error_is_string():
    entry = {"id": "curvature", "sha256": "c" * 64}
    result = checksum_mismatch_error(entry, "/path/model.pth", "d" * 64)
    assert isinstance(result, str)
