"""
Tests for ZebrafishAnalysisMainWidget._ensure_models_ready().

The method must:
- Return True immediately when slicer.app.testingEnabled() is True (no dialog, no net).
- Return True immediately when all required model files are already cached.
- Show a QMessageBox and return False when the user declines download.
- Show a QMessageBox and delegate to download_models() when user confirms.
- Return download_models() result (True on success, False on failure).
- Only require the eye model when the _chk_eyes checkbox is checked.

No real Slicer/Qt/VTK environment needed — all Slicer/Qt symbols are mocked.
"""

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Context manager: inject stub modules, restore sys.modules exactly on exit.
# This prevents module identity splits that would break other test files.
# ---------------------------------------------------------------------------

@contextmanager
def _stub_slicer_env(slicer_mock):
    """Inject Slicer/Qt/CTK/VTK stubs and a fresh ZebrafishAnalysisLib.widget.

    Restores sys.modules exactly as found on entry so other test files that
    hold top-level references to ZebrafishAnalysisLib.model_manifest etc.
    are not affected.
    """
    qt_mock = MagicMock()
    qt_mock.QMessageBox.Yes = 0x4000
    qt_mock.QMessageBox.No = 0x10000
    msg_instance = MagicMock()
    qt_mock.QMessageBox.return_value = msg_instance

    stub_names = [
        "slicer",
        "qt",
        "ctk",
        "vtk",
        "slicer.ScriptedLoadableModule",
        "slicer.util",
    ]
    lib_names = [k for k in sys.modules if "ZebrafishAnalysisLib" in k
                                         or "ZebrafishAnalysisCore" in k]

    # Snapshot state we will restore.
    saved = {k: sys.modules[k] for k in stub_names + lib_names if k in sys.modules}

    # Inject stubs.
    sys.modules["slicer"] = slicer_mock
    sys.modules["qt"] = qt_mock
    sys.modules["ctk"] = MagicMock()
    sys.modules["vtk"] = MagicMock()
    sys.modules["slicer.ScriptedLoadableModule"] = MagicMock()
    sys.modules["slicer.util"] = MagicMock()

    # Evict widget so it re-imports under the stubs.
    widget_key = "ZebrafishAnalysisLib.widget"
    sys.modules.pop(widget_key, None)

    try:
        yield qt_mock, msg_instance
    finally:
        # Remove only what we added (stubs + widget).
        for k in stub_names + [widget_key]:
            sys.modules.pop(k, None)
        # Restore previously existing modules.
        sys.modules.update(saved)


def _make_slicer_mock(testing_enabled=False):
    mock = MagicMock()
    mock.app.testingEnabled.return_value = testing_enabled
    return mock


def _make_checkbox(checked=False):
    chk = MagicMock()
    chk.isChecked.return_value = checked
    return chk


def _run_ensure(slicer_mock, get_missing_mock, download_mock,
                user_answer, model_id="general", eyes_checked=False,
                curvature_checked=True):
    """
    Call _ensure_models_ready with mocked dependencies; return the bool result.

    user_answer should be qt_mock.QMessageBox.Yes (0x4000) or .No (0x10000).
    eyes_checked controls whether the _chk_eyes checkbox is ticked.
    curvature_checked controls whether the _chk_curvature checkbox is ticked.
    """
    with _stub_slicer_env(slicer_mock) as (qt_mock, msg_instance):
        msg_instance.exec_.return_value = user_answer

        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget
        import ZebrafishAnalysisLib.model_manifest as _mm
        import ZebrafishAnalysisLib.model_downloader as _md

        with patch.object(_mm, "get_missing_models", get_missing_mock), \
             patch.object(_md, "download_models", download_mock):

            widget = object.__new__(ZebrafishAnalysisMainWidget)
            widget._chk_eyes = _make_checkbox(checked=eyes_checked)
            widget._chk_curvature = _make_checkbox(checked=curvature_checked)
            return widget._ensure_models_ready(model_id)


# ---------------------------------------------------------------------------
# Test: testingEnabled=True → immediate True, no dialog, no download
# ---------------------------------------------------------------------------

def test_ensure_models_ready_testing_enabled_returns_true():
    slicer_mock = _make_slicer_mock(testing_enabled=True)
    get_missing_mock = MagicMock(return_value=[{"label": "General body model"}])
    download_mock = MagicMock(return_value=False)

    result = _run_ensure(slicer_mock, get_missing_mock, download_mock,
                         user_answer=0x4000)

    assert result is True
    download_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Test: all models cached → True, no dialog, no download
# ---------------------------------------------------------------------------

def test_ensure_models_ready_all_cached_returns_true():
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    get_missing_mock = MagicMock(return_value=[])  # nothing missing
    download_mock = MagicMock(return_value=False)

    result = _run_ensure(slicer_mock, get_missing_mock, download_mock,
                         user_answer=0x10000, eyes_checked=False)

    assert result is True
    download_mock.assert_not_called()
    # get_missing_models called with body+curvature only (no eye key)
    called_dict = get_missing_mock.call_args[0][0]
    assert "body" in called_dict
    assert "curvature" in called_dict
    assert "eye" not in called_dict


# ---------------------------------------------------------------------------
# Test: user declines → False, download not called
# ---------------------------------------------------------------------------

def test_ensure_models_ready_user_declines_returns_false():
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    missing_entries = [{"label": "General body model"}]
    get_missing_mock = MagicMock(return_value=missing_entries)
    download_mock = MagicMock(return_value=True)

    result = _run_ensure(slicer_mock, get_missing_mock, download_mock,
                         user_answer=0x10000)  # No

    assert result is False
    download_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Test: user accepts + download succeeds → True
# ---------------------------------------------------------------------------

def test_ensure_models_ready_user_accepts_download_succeeds():
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    missing_entries = [{"label": "General body model"}]
    get_missing_mock = MagicMock(return_value=missing_entries)
    download_mock = MagicMock(return_value=True)

    result = _run_ensure(slicer_mock, get_missing_mock, download_mock,
                         user_answer=0x4000)  # Yes

    assert result is True
    download_mock.assert_called_once_with(missing_entries)


# ---------------------------------------------------------------------------
# Test: user accepts + download fails → False
# ---------------------------------------------------------------------------

def test_ensure_models_ready_user_accepts_download_fails():
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    missing_entries = [{"label": "General body model"}]
    get_missing_mock = MagicMock(return_value=missing_entries)
    download_mock = MagicMock(return_value=False)

    result = _run_ensure(slicer_mock, get_missing_mock, download_mock,
                         user_answer=0x4000)  # Yes

    assert result is False
    download_mock.assert_called_once_with(missing_entries)


# ---------------------------------------------------------------------------
# Test: eyes unchecked → eye model excluded from missing check even if absent
# ---------------------------------------------------------------------------

def test_ensure_models_ready_eyes_unchecked_eye_model_not_required():
    """When eyes checkbox is off, eye model must not appear in the required set."""
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    get_missing_mock = MagicMock(return_value=[])
    download_mock = MagicMock(return_value=False)

    _run_ensure(slicer_mock, get_missing_mock, download_mock,
                user_answer=0x10000, eyes_checked=False)

    called_dict = get_missing_mock.call_args[0][0]
    assert "eye" not in called_dict, (
        "eye model must not be in required set when eyes checkbox is unchecked"
    )


# ---------------------------------------------------------------------------
# Test: eyes checked → eye model included in missing check
# ---------------------------------------------------------------------------

def test_ensure_models_ready_eyes_checked_eye_model_required():
    """When eyes checkbox is on, eye model must appear in the required set."""
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    get_missing_mock = MagicMock(return_value=[])
    download_mock = MagicMock(return_value=False)

    _run_ensure(slicer_mock, get_missing_mock, download_mock,
                user_answer=0x10000, eyes_checked=True)

    called_dict = get_missing_mock.call_args[0][0]
    assert "eye" in called_dict, (
        "eye model must be in required set when eyes checkbox is checked"
    )
    assert "body" in called_dict
    assert "curvature" in called_dict


# ---------------------------------------------------------------------------
# Test: curvature unchecked → curvature model excluded from required set
# ---------------------------------------------------------------------------

def test_ensure_models_ready_curvature_unchecked_not_required():
    """When curvature checkbox is off, curvature model must not appear in required set."""
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    get_missing_mock = MagicMock(return_value=[])
    download_mock = MagicMock(return_value=False)

    _run_ensure(slicer_mock, get_missing_mock, download_mock,
                user_answer=0x10000, curvature_checked=False)

    called_dict = get_missing_mock.call_args[0][0]
    assert "curvature" not in called_dict, (
        "curvature model must not be in required set when curvature checkbox is unchecked"
    )
    assert "body" in called_dict


# ---------------------------------------------------------------------------
# Test: curvature checked → curvature model included in required set
# ---------------------------------------------------------------------------

def test_ensure_models_ready_curvature_checked_is_required():
    """When curvature checkbox is on, curvature model must appear in required set."""
    slicer_mock = _make_slicer_mock(testing_enabled=False)
    get_missing_mock = MagicMock(return_value=[])
    download_mock = MagicMock(return_value=False)

    _run_ensure(slicer_mock, get_missing_mock, download_mock,
                user_answer=0x10000, curvature_checked=True)

    called_dict = get_missing_mock.call_args[0][0]
    assert "curvature" in called_dict, (
        "curvature model must be in required set when curvature checkbox is checked"
    )
    assert "body" in called_dict
