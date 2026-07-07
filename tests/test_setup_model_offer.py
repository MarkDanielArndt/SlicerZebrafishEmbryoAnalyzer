"""
Tests for the initial setup model-offer helpers.

Pure Python — no slicer, qt, vtk, ctk, or torch required.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_MODULE_DIR = Path(__file__).resolve().parent.parent / "ZebrafishEmbryoAnalyzer"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from ZebrafishEmbryoAnalyzerLib.model_manifest import (
    MODEL_SETS,
    collect_all_model_entries,
    get_missing_models,
)


def test_collect_all_entries_no_duplicates():
    """collect_all_model_entries() result has no duplicate ids."""
    result = collect_all_model_entries()
    ids = list(result.keys())
    assert len(ids) == len(set(ids))


def test_collect_all_entries_covers_all_sets():
    """Every entry in every MODEL_SET is present in collect_all_model_entries()."""
    result = collect_all_model_entries()
    for variant_name, variant in MODEL_SETS.items():
        for role, entry in variant.items():
            assert entry["id"] in result, (
                f"Entry id={entry['id']!r} from MODEL_SETS[{variant_name!r}][{role!r}] "
                f"missing from collect_all_model_entries()"
            )


def test_missing_excludes_cached(tmp_path):
    """get_missing_models() excludes entries whose cache path exists with content."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as mm

    all_entries = collect_all_model_entries()
    # Pick two entries to mark as cached
    entry_ids = list(all_entries.keys())
    cached_ids = set(entry_ids[:2])

    orig = mm._CACHE_DIR
    mm._CACHE_DIR = tmp_path
    try:
        for eid in cached_ids:
            entry = all_entries[eid]
            p = tmp_path / entry["filename"]
            p.write_bytes(b"dummy weights")

        missing = get_missing_models(all_entries)
    finally:
        mm._CACHE_DIR = orig

    missing_ids = {e["id"] for e in missing}
    for eid in cached_ids:
        assert eid not in missing_ids, (
            f"Cached entry {eid!r} should not appear in missing list"
        )


def test_missing_includes_zero_byte_file():
    """get_missing_models() treats a zero-byte file as missing."""
    from unittest.mock import MagicMock

    all_entries = collect_all_model_entries()
    first_id = next(iter(all_entries))
    target_entry = all_entries[first_id]

    stat_result = MagicMock()
    stat_result.st_size = 0

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat", return_value=stat_result):
        missing = get_missing_models({first_id: target_entry})

    assert any(e["id"] == first_id for e in missing), (
        f"Zero-byte entry {first_id!r} should appear in missing list"
    )


def test_missing_empty_when_all_cached(tmp_path):
    """get_missing_models() returns [] when all paths exist with content."""
    import ZebrafishEmbryoAnalyzerLib.model_manifest as mm

    all_entries = collect_all_model_entries()

    orig = mm._CACHE_DIR
    mm._CACHE_DIR = tmp_path
    try:
        for entry in all_entries.values():
            p = tmp_path / entry["filename"]
            p.write_bytes(b"dummy weights")

        missing = get_missing_models(all_entries)
    finally:
        mm._CACHE_DIR = orig

    assert missing == []
