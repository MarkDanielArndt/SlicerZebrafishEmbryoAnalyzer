"""
Tests for ZebrafishAnalysisLib.model_downloader.

The test environment has no slicer/qt/vtk/ctk.  Tests cover:
- testingEnabled guard: download_models returns True immediately in test mode
- _run_downloads: pure-Python download worker (mocked requests)
"""

import os
import sys
import threading
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Guard: download_models returns True when slicer.app.testingEnabled() is True
# ---------------------------------------------------------------------------

def test_download_models_returns_true_in_testing_mode(monkeypatch):
    """download_models must return True immediately when testing mode active."""
    mock_slicer = MagicMock()
    mock_slicer.app.testingEnabled.return_value = True
    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setitem(sys.modules, "qt", MagicMock())

    from ZebrafishAnalysisLib.model_downloader import download_models
    from ZebrafishAnalysisLib.model_manifest import MODELS

    result = download_models([MODELS["general_body"]])
    assert result is True
    # Must not call mainWindow — guard fires before any dialog.
    mock_slicer.util.mainWindow.assert_not_called()


def test_download_models_returns_true_for_empty_list(monkeypatch):
    """download_models returns True immediately when given an empty list."""
    mock_slicer = MagicMock()
    mock_slicer.app.testingEnabled.return_value = True
    monkeypatch.setitem(sys.modules, "slicer", mock_slicer)
    monkeypatch.setitem(sys.modules, "qt", MagicMock())

    from ZebrafishAnalysisLib.model_downloader import download_models

    result = download_models([])
    assert result is True


def test_download_models_returns_true_when_slicer_absent(monkeypatch):
    """When slicer is not importable, download_models returns True (non-Slicer env)."""
    # Remove slicer from modules so ImportError fires inside download_models.
    monkeypatch.delitem(sys.modules, "slicer", raising=False)

    # Temporarily make 'import slicer' raise ImportError.
    import builtins
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "slicer":
            raise ImportError("slicer not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ZebrafishAnalysisLib.model_downloader import download_models
    from ZebrafishAnalysisLib.model_manifest import MODELS

    result = download_models([MODELS["general_body"]])
    assert result is True


# ---------------------------------------------------------------------------
# _run_downloads: worker logic (mocked requests, no network)
# ---------------------------------------------------------------------------

def _make_progress_state(total_bytes=0):
    """Return a fresh progress_state dict matching _run_downloads expectations."""
    return {
        "downloaded_bytes": 0,
        "total_bytes": total_bytes,
        "current_label": "",
        "done_flag": False,
        "cancelled": False,
        "error": None,
    }


def test_run_downloads_marks_done_on_success(tmp_path):
    """Worker sets done_flag=True after downloading all entries."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    content = b"fake model weights"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {"content-length": str(len(content))}
    mock_response.iter_content.return_value = [content]

    mock_head = MagicMock()
    mock_head.headers = {"content-length": str(len(content))}

    progress_state = _make_progress_state()
    cancel_event = threading.Event()

    url = "https://huggingface.co/fake/model.pth"
    filename = "model.pth"

    with patch("requests.get", return_value=mock_response), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [(url, filename, "test model", 555_000_000)],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["done_flag"] is True
    assert progress_state["error"] is None
    assert progress_state["cancelled"] is False
    # File must exist at final path.
    assert (tmp_path / filename).exists()
    # downloaded_bytes must reflect actual bytes written.
    assert progress_state["downloaded_bytes"] == len(content)


def test_run_downloads_updates_current_label(tmp_path):
    """Worker updates current_label as it moves through models."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    content = b"weights"
    labels_seen = []

    def fake_iter_content(chunk_size=None):
        labels_seen.append(progress_state["current_label"])
        yield content

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {}
    mock_response.iter_content = fake_iter_content

    mock_head = MagicMock()
    mock_head.headers = {}

    progress_state = _make_progress_state()
    cancel_event = threading.Event()

    with patch("requests.get", return_value=mock_response), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [
                ("https://example.com/a.pth", "a.pth", "Model A", 100),
                ("https://example.com/b.pth", "b.pth", "Model B", 100),
            ],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["done_flag"] is True
    assert "Model A" in labels_seen
    assert "Model B" in labels_seen or progress_state["current_label"] == "Model B"


def test_run_downloads_accumulates_downloaded_bytes(tmp_path):
    """downloaded_bytes is cumulative across multiple models."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    content_a = b"aaaa"  # 4 bytes
    content_b = b"bbbbbb"  # 6 bytes

    call_count = {"n": 0}

    def fake_get(url, **kwargs):
        call_count["n"] += 1
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.headers = {}
        if call_count["n"] == 1:
            resp.iter_content.return_value = [content_a]
        else:
            resp.iter_content.return_value = [content_b]
        return resp

    mock_head = MagicMock()
    mock_head.headers = {}

    progress_state = _make_progress_state()
    cancel_event = threading.Event()

    with patch("requests.get", side_effect=fake_get), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [
                ("https://example.com/a.pth", "a.pth", "A", 0),
                ("https://example.com/b.pth", "b.pth", "B", 0),
            ],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["done_flag"] is True
    assert progress_state["downloaded_bytes"] == len(content_a) + len(content_b)


def test_run_downloads_total_bytes_updated_from_content_length(tmp_path):
    """total_bytes is updated when Content-Length header is available."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    content = b"x" * 1000

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {}
    mock_response.iter_content.return_value = [content]

    mock_head = MagicMock()
    mock_head.headers = {"content-length": "1000"}

    # Start with 0 estimate (unknown size).
    progress_state = _make_progress_state(total_bytes=0)
    cancel_event = threading.Event()

    with patch("requests.get", return_value=mock_response), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [("https://example.com/m.pth", "m.pth", "Model", 0)],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["total_bytes"] == 1000


def test_run_downloads_cancels_mid_stream(tmp_path):
    """Worker stops and sets cancelled=True when cancel_event fires."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    cancel_event = threading.Event()

    # Simulate cancellation by firing the event on first chunk iteration.
    def _iter_content(chunk_size=None):
        cancel_event.set()
        yield b"partial data"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {}
    mock_response.iter_content = _iter_content

    mock_head = MagicMock()
    mock_head.headers = {}

    progress_state = _make_progress_state()

    with patch("requests.get", return_value=mock_response), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [("https://example.com/model.pth", "model.pth", "test", 0)],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["cancelled"] is True
    assert progress_state["done_flag"] is False
    # Temporary file must be cleaned up.
    assert not (tmp_path / "model.pth.tmp").exists()


def test_run_downloads_records_error_on_http_failure(tmp_path):
    """Worker records error message when HTTP request fails."""
    from ZebrafishAnalysisLib.model_downloader import _run_downloads

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")

    mock_head = MagicMock()
    mock_head.headers = {}

    progress_state = _make_progress_state()
    cancel_event = threading.Event()

    with patch("requests.get", return_value=mock_response), \
         patch("requests.head", return_value=mock_head), \
         patch("ZebrafishAnalysisLib.model_downloader._CACHE_DIR", tmp_path):
        _run_downloads(
            [("https://example.com/model.pth", "model.pth", "test", 0)],
            {},
            progress_state,
            cancel_event,
        )

    assert progress_state["error"] is not None
    assert "404" in progress_state["error"]
    assert progress_state["done_flag"] is False
